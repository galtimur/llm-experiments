import torch


def track_params(model, model_start, model_previous):
    num_el = dist = grad_step = dev_step = sum_step = sum_dist = dev_dist = 0
    for (name, param), (_, param_start), (_, param_prev) in zip(
        model.named_parameters(),
        model_start.named_parameters(),
        model_previous.named_parameters(),
    ):
        dist_tensor = param.data - param_start.data
        step = param.data - param_prev.data

        dist += torch.norm(dist_tensor) ** 2
        grad_step += torch.norm(step) ** 2
        num_el += param.data.numel()

        sum_step += torch.sum(step)
        dev_step += torch.sum((step - sum_step / num_el) ** 2)

        sum_dist += torch.sum(dist_tensor)
        dev_dist += torch.sum((dist_tensor - sum_dist / num_el) ** 2)
        # mean_dist += torch.sum(dist_tensor)
        # param_abs = torch.abs(param.grad.data)
        # grad_norm = torch.mean(param_abs)
        # grad_max = torch.max(param_abs)
        return (
            torch.sqrt(dist / num_el),
            torch.sqrt(grad_step / num_el),
            torch.sqrt(dev_step / num_el),
            torch.sqrt(dev_dist / num_el),
        )


def filter_grad(model, mask_dict, threshold, type, apply_saved_mask):
    num_el = 0
    num_grad = 0
    num_common = 0
    for name, param in model.named_parameters():
        grad_norm2 = torch.norm(param.grad.data)
        num_el += param.grad.data.numel()
        if not apply_saved_mask:
            param_abs = torch.abs(param.grad.data)
            grad_norm = torch.mean(param_abs)
            grad_max = torch.max(param_abs)
            grad_min = torch.min(param_abs)
            if type == "largest":
                thresh = grad_norm / threshold
                if thresh > grad_max:
                    thresh = (1 - threshold) * grad_max
                mask = param_abs > thresh
            if type == "smallest":
                thresh = grad_norm * threshold
                if thresh < grad_min:
                    thresh = (1 + threshold) * grad_min
                mask = param_abs < thresh
        else:
            mask = mask_dict[name]
        if torch.sum(mask) > 0:
            param.grad.data = param.grad.data * mask.float()
            # param.grad.data = (
            #     param.grad.data
            #     / (torch.norm(param.grad.data) + 0.1 * grad_norm2)
            #     * grad_norm2
            # )
            if name in mask_dict:
                common_mask = mask_dict[name] & mask
                num_common += torch.sum(common_mask)
            num_grad += torch.sum(mask)
            mask_dict[name] = mask

    return num_grad / num_el, num_common / num_grad, mask_dict
