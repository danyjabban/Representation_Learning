import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), 0., 1.)
    # Return perturbed samples
    return x_adv, None

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat (data) and lbl (label) are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    if rand_start:
        x_nat_perturbed = x_nat + (torch.rand(x_nat.shape, device=device) * (2*eps) - eps)
        # x_nat_perturbed = (x_nat_perturbed - x_nat_perturbed.min()) / x_nat_perturbed.max()
        x_nat_perturbed = torch.clamp(x_nat_perturbed, min=0, max=1)
        # Make sure the sample is projected into original distribution bounds [0,1]
    else:
        x_nat_perturbed = torch.clone(x_nat)

    # Iterate over iters
    for iii in range(int(iters)):
        # Compute gradient w.r.t. data (we give you this function, but understand it)
        grad_wrt_data = gradient_wrt_data(model, device, data=x_nat_perturbed, lbl=lbl)
        # Perturb the image using the gradient
        x_nat_perturbed += torch.sign(grad_wrt_data) * alpha
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        x_nat_perturbed = torch.clamp(x_nat_perturbed, min=x_nat-eps, max=x_nat+eps)
        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
        x_nat_perturbed = torch.clamp(x_nat_perturbed, min=0, max=1)

    # Return the final perturbed samples
    assert(torch.max(torch.abs(x_nat_perturbed-x_nat)) <= (eps + 1e-7)), \
        "torch.max(torch.abs(x_nat_perturbed-x_nat))=%.10f,eps=%.10f" % (torch.max(torch.abs(x_nat_perturbed-x_nat)), eps)
    assert(x_nat_perturbed.max() == 1.), "x_nat_perturbed.max()=%.10f,eps=%.10f" % (x_nat_perturbed.max(), eps)
    assert(x_nat_perturbed.min() == 0.), "x_nat_perturbed.min()=%.10f,eps=%.10f" % (x_nat_perturbed.min(), eps)
    return x_nat_perturbed, lbl


def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    x_adv, lbl = PGD_attack(model=model, device=device, dat=dat, lbl=lbl, eps=eps, alpha=eps, iters=1, rand_start=False)
    # HINT: FGSM is a special case of PGD
    return x_adv, lbl


def rFGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    x_adv, lbl = PGD_attack(model=model, device=device, dat=dat, lbl=lbl, eps=eps, alpha=eps, iters=1, rand_start=True)
    # HINT: rFGSM is a special case of PGD
    return x_adv, lbl


def FGM_L2_attack(model, device, dat, lbl, eps):
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # Compute gradient w.r.t. data
    grad_wrt_data = gradient_wrt_data(model, device, data=x_nat.to(device), lbl=lbl.to(device))
    # shape: [64, 1, 28, 28]

    # Compute sample-wise L2 norm of gradient (L2 norm for each batch element)
    # HINT: Flatten gradient tensor first, then compute L2 norm
    # grad_norm = torch.norm(grad_wrt_data, p=2, dim=(2, 3), keepdim=True) # shape should be [64, 1, 1, 1]
    grad_norm = torch.flatten(grad_wrt_data, 2, 3).norm(p=2, dim=2)

    # Perturb the data using the gradient
    # HINT: Before normalizing the gradient by its L2 norm, use
    # torch.clamp(l2_of_grad, min=1e-12) to prevent division by 0
    x_adv = x_nat + eps * (grad_wrt_data / torch.clamp(grad_norm, min=1e-12))
    # Add perturbation the data

    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv, min=0, max=1)
    # Return the perturbed samples
    return x_adv, lbl
