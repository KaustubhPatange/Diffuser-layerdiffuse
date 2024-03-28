import torch

import lib_layerdiffusion.utils


class UnetPatcher:
    def __init__(self, model, offload_device):
        model_sd = model.state_dict()
        self.model = model
        self.model_keys = set(model_sd.keys())
        self.patches = {}
        self.backup = {}
        self.offload_device = offload_device

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        p = set()
        for k in patches:
            if k in self.model_keys:
                p.add(k)
                current_patches = self.patches.get(k, [])
                current_patches.append((strength_patch, patches[k], strength_model))
                self.patches[k] = current_patches

        return list(p)

    def load_frozen_patcher(self, state_dict, strength):
        patch_dict = {}
        for k, w in state_dict.items():
            model_key, patch_type, weight_index = k.split("::")
            if model_key not in patch_dict:
                patch_dict[model_key] = {}
            if patch_type not in patch_dict[model_key]:
                patch_dict[model_key][patch_type] = [None] * 16
            patch_dict[model_key][patch_type][int(weight_index)] = w

        patch_flat = {}
        for model_key, v in patch_dict.items():
            for patch_type, weight_list in v.items():
                patch_flat[model_key] = (patch_type, weight_list)

        self.add_patches(
            patches=patch_flat, strength_patch=float(strength), strength_model=1.0
        )
        return

    def model_state_dict(self, filter_prefix=None):
        sd = self.model.state_dict()
        keys = list(sd.keys())
        if filter_prefix is not None:
            for k in keys:
                if not k.startswith(filter_prefix):
                    sd.pop(k)
        return sd

    def patch_model(self, device_to=None, patch_weights=True):
        if patch_weights:
            model_sd = self.model_state_dict()
            for key in self.patches:
                if key not in model_sd:
                    print("could not patch. key doesn't exist in model:", key)
                    continue

                weight = model_sd[key]

                inplace_update = True  # condition? maybe

                if key not in self.backup:
                    self.backup[key] = weight.to(
                        device=self.offload_device, copy=inplace_update
                    )

                if device_to is not None:
                    temp_weight = lib_layerdiffusion.utils.cast_to_device(
                        weight, device_to, torch.float32, copy=True
                    )
                else:
                    temp_weight = weight.to(torch.float32, copy=True)
                out_weight = self.calculate_weight(
                    self.patches[key], temp_weight, key
                ).to(weight.dtype)
                if inplace_update:
                    lib_layerdiffusion.utils.copy_to_param(self.model, key, out_weight)
                else:
                    lib_layerdiffusion.utils.set_attr(self.model, key, out_weight)
                del temp_weight

            if device_to is not None:
                self.model.to(device_to)
                self.current_device = device_to

        return self.model

    def calculate_weight(self, patches, weight, key):
        for p in patches:
            alpha = p[0]
            v = p[1]
            strength_model = p[2]

            if strength_model != 1.0:
                weight *= strength_model

            if isinstance(v, list):
                v = (self.calculate_weight(v[1:], v[0].clone(), key),)

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]
            else:
                raise Exception("Could not detect patch_type")

            if patch_type == "diff":
                w1 = v[0]
                if alpha != 0.0:
                    if w1.shape != weight.shape:
                        if w1.ndim == weight.ndim == 4:
                            new_shape = [
                                max(n, m) for n, m in zip(weight.shape, w1.shape)
                            ]
                            print(f"Merged with {key} channel changed to {new_shape}")
                            new_diff = alpha * lib_layerdiffusion.utils.cast_to_device(
                                w1, weight.device, weight.dtype
                            )
                            new_weight = torch.zeros(size=new_shape).to(weight)
                            new_weight[
                                : weight.shape[0],
                                : weight.shape[1],
                                : weight.shape[2],
                                : weight.shape[3],
                            ] = weight
                            new_weight[
                                : new_diff.shape[0],
                                : new_diff.shape[1],
                                : new_diff.shape[2],
                                : new_diff.shape[3],
                            ] += new_diff
                            new_weight = new_weight.contiguous().clone()
                            weight = new_weight
                        else:
                            print(
                                "WARNING SHAPE MISMATCH {} WEIGHT NOT MERGED {} != {}".format(
                                    key, w1.shape, weight.shape
                                )
                            )
                    else:
                        weight += alpha * lib_layerdiffusion.utils.cast_to_device(
                            w1, weight.device, weight.dtype
                        )
            elif patch_type == "lora":  # lora/locon
                mat1 = lib_layerdiffusion.utils.cast_to_device(
                    v[0], weight.device, torch.float32
                )
                mat2 = lib_layerdiffusion.utils.cast_to_device(
                    v[1], weight.device, torch.float32
                )
                if v[2] is not None:
                    alpha *= v[2] / mat2.shape[0]
                if v[3] is not None:
                    # locon mid weights, hopefully the math is fine because I didn't properly test it
                    mat3 = lib_layerdiffusion.utils.cast_to_device(
                        v[3], weight.device, torch.float32
                    )
                    final_shape = [
                        mat2.shape[1],
                        mat2.shape[0],
                        mat3.shape[2],
                        mat3.shape[3],
                    ]
                    mat2 = (
                        torch.mm(
                            mat2.transpose(0, 1).flatten(start_dim=1),
                            mat3.transpose(0, 1).flatten(start_dim=1),
                        )
                        .reshape(final_shape)
                        .transpose(0, 1)
                    )
                try:
                    weight += (
                        (
                            alpha
                            * torch.mm(
                                mat1.flatten(start_dim=1), mat2.flatten(start_dim=1)
                            )
                        )
                        .reshape(weight.shape)
                        .type(weight.dtype)
                    )
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "lokr":
                w1 = v[0]
                w2 = v[1]
                w1_a = v[3]
                w1_b = v[4]
                w2_a = v[5]
                w2_b = v[6]
                t2 = v[7]
                dim = None

                if w1 is None:
                    dim = w1_b.shape[0]
                    w1 = torch.mm(
                        lib_layerdiffusion.utils.cast_to_device(
                            w1_a, weight.device, torch.float32
                        ),
                        lib_layerdiffusion.utils.cast_to_device(
                            w1_b, weight.device, torch.float32
                        ),
                    )
                else:
                    w1 = lib_layerdiffusion.utils.cast_to_device(
                        w1, weight.device, torch.float32
                    )

                if w2 is None:
                    dim = w2_b.shape[0]
                    if t2 is None:
                        w2 = torch.mm(
                            lib_layerdiffusion.utils.cast_to_device(
                                w2_a, weight.device, torch.float32
                            ),
                            lib_layerdiffusion.utils.cast_to_device(
                                w2_b, weight.device, torch.float32
                            ),
                        )
                    else:
                        w2 = torch.einsum(
                            "i j k l, j r, i p -> p r k l",
                            lib_layerdiffusion.utils.cast_to_device(
                                t2, weight.device, torch.float32
                            ),
                            lib_layerdiffusion.utils.cast_to_device(
                                w2_b, weight.device, torch.float32
                            ),
                            lib_layerdiffusion.utils.cast_to_device(
                                w2_a, weight.device, torch.float32
                            ),
                        )
                else:
                    w2 = lib_layerdiffusion.utils.cast_to_device(
                        w2, weight.device, torch.float32
                    )

                if len(w2.shape) == 4:
                    w1 = w1.unsqueeze(2).unsqueeze(2)
                if v[2] is not None and dim is not None:
                    alpha *= v[2] / dim

                try:
                    weight += alpha * torch.kron(w1, w2).reshape(weight.shape).type(
                        weight.dtype
                    )
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "loha":
                w1a = v[0]
                w1b = v[1]
                if v[2] is not None:
                    alpha *= v[2] / w1b.shape[0]
                w2a = v[3]
                w2b = v[4]
                if v[5] is not None:  # cp decomposition
                    t1 = v[5]
                    t2 = v[6]
                    m1 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        lib_layerdiffusion.utils.cast_to_device(
                            t1, weight.device, torch.float32
                        ),
                        lib_layerdiffusion.utils.cast_to_device(
                            w1b, weight.device, torch.float32
                        ),
                        lib_layerdiffusion.utils.cast_to_device(
                            w1a, weight.device, torch.float32
                        ),
                    )

                    m2 = torch.einsum(
                        "i j k l, j r, i p -> p r k l",
                        lib_layerdiffusion.utils.cast_to_device(
                            t2, weight.device, torch.float32
                        ),
                        lib_layerdiffusion.utils.cast_to_device(
                            w2b, weight.device, torch.float32
                        ),
                        lib_layerdiffusion.utils.cast_to_device(
                            w2a, weight.device, torch.float32
                        ),
                    )
                else:
                    m1 = torch.mm(
                        lib_layerdiffusion.utils.cast_to_device(
                            w1a, weight.device, torch.float32
                        ),
                        lib_layerdiffusion.utils.cast_to_device(
                            w1b, weight.device, torch.float32
                        ),
                    )
                    m2 = torch.mm(
                        lib_layerdiffusion.utils.cast_to_device(
                            w2a, weight.device, torch.float32
                        ),
                        lib_layerdiffusion.utils.cast_to_device(
                            w2b, weight.device, torch.float32
                        ),
                    )

                try:
                    weight += (alpha * m1 * m2).reshape(weight.shape).type(weight.dtype)
                except Exception as e:
                    print("ERROR", key, e)
            elif patch_type == "glora":
                if v[4] is not None:
                    alpha *= v[4] / v[0].shape[0]

                a1 = lib_layerdiffusion.utils.cast_to_device(
                    v[0].flatten(start_dim=1), weight.device, torch.float32
                )
                a2 = lib_layerdiffusion.utils.cast_to_device(
                    v[1].flatten(start_dim=1), weight.device, torch.float32
                )
                b1 = lib_layerdiffusion.utils.cast_to_device(
                    v[2].flatten(start_dim=1), weight.device, torch.float32
                )
                b2 = lib_layerdiffusion.utils.cast_to_device(
                    v[3].flatten(start_dim=1), weight.device, torch.float32
                )

                weight += (
                    (
                        (
                            torch.mm(b2, b1)
                            + torch.mm(torch.mm(weight.flatten(start_dim=1), a2), a1)
                        )
                        * alpha
                    )
                    .reshape(weight.shape)
                    .type(weight.dtype)
                )
            # elif patch_type in extra_weight_calculators:
            #     weight = extra_weight_calculators[patch_type](weight, alpha, v)
            else:
                print("patch type not recognized", patch_type, key)

        return weight
