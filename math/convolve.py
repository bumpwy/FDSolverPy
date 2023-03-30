#!/usr/bin/env python
import jax.numpy as jnp
import jax

def pad_to_shape(a,shape):
    ns_, ns = shape, a.shape
    ns_pad = [(n_ - n) for n_,n in zip(ns_,ns)]
    pads = tuple((n_pad//2,n_pad//2+n_pad%2) for n_pad in ns_pad)

    if len(ns)==1:
        a_pad = jnp.pad(a,(ns_pad[0]//2,ns_pad[0]//2+ns_pad[0]%2),mode='constant')
    else:
        a_pad = jnp.pad(a,pads,mode='constant')

    return jnp.fft.ifftshift(a_pad)
@jax.jit
def convolve_fft(A,f):
    Ak = jnp.fft.rfftn(A)
    fe = pad_to_shape(f,A.shape)
    fek = jnp.fft.rfftn(fe)

    return jnp.fft.irfftn(Ak*fek)

