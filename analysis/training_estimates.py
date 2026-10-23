GB=1e9
models={'3B':(3e9,2048,36),'8B':(8e9,4096,36),'14B':(14e9,5120,48),'30B':(30e9,5120,64)}
s=4096;b=1
def act(h,L): return (2*s*b*h*L + 34*s*b*h)/GB
for name,(P,h,L) in models.items():
    lora=0.006*P*16/GB
    a=act(h,L)
    ov=1.5
    q=0.55*P/GB+lora+a+ov
    l=2*P/GB+lora+a+ov
    f=16*P/GB
    print(f"{name}: act(ckpt)={a:.1f}GB loraStates={lora:.2f}GB QLoRA~{q:.0f}GB LoRA~{l:.0f}GB FullFT states={f:.0f}GB (+act) -> per GPU with ZeRO3 on 8: {f/8+a+ov:.0f}GB")
print()
gpus={'RTX3090':(71e12,24,0.22),'RTX4090':(165e12,24,0.34),'RTX5090':(209.5e12,32,0.69),'L4':(121e12,24,0.44),'A100-40':(312e12,40,0.89),'A100-80':(312e12,80,1.19),'H100':(989e12,80,2.69),'H200':(989e12,141,3.59)}
D=90e6
for name,(P,h,L) in models.items():
    for g,(fl,mem,price) in gpus.items():
        for meth,mfu,factor in [('QLoRA',0.25,6),('LoRA',0.35,6)]:
            C=factor*P*D
            t=C/(fl*mfu)/3600
            print(f"{name} {meth} on {g}: {t:7.1f} h  ${t*price:7.2f}")
    print()
