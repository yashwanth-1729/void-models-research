FX=95
def fmt(x):
    # rupees in lakh
    return f"{x/1e5:,.1f}L"
print("=== India SMB vertical SaaS (doc workflow), Rs ===")
price=1500; docs=2000
inf_doc_usd=0.9*0.00055+0.1*0.011
inf=docs*inf_doc_usd*FX
print("inference per cust-month Rs", round(inf), "per doc Rs", round(inf_doc_usd*FX,3))
support=250; payfee=0.0236; churn=0.04; cac=6000
for N,fixed_infra,other_opex,team in [(10,5000*12,2e5,0),(100,15000*12,6e5,0.5e5*12),(1000,80000*12,30e5,(6*1e5+1.5e5)*12)]:
    rev=N*price*12
    cogs=N*12*(inf+support+50)+fixed_infra+rev*payfee
    gp=rev-cogs
    cacspend=N*churn*12*cac
    op=gp-cacspend-other_opex
    print(f"N={N}: revenue {fmt(rev)} cogs {fmt(cogs)} GP {fmt(gp)} GM {gp/rev*100:.0f}% CAC/yr {fmt(cacspend)} other opex {fmt(other_opex)} OpProfit pre-comp {fmt(op)} team+founder comp {fmt(team)} after {fmt(op-team)}; one-time CAC to reach N {fmt(N*cac)}")
print()
print("=== Global dev-tool SaaS (CI-fix), $ -> Rs ===")
price=49; inf=14; support=3; payfee=0.05; churn=0.03; cac=150
for N,fixed_infra,other_opex,team in [(10,200*12,2e5/FX,0),(100,600*12,6e5/FX,0.5e5*12/FX),(1000,4000*12,30e5/FX,(6*1e5+1.5e5)*12/FX)]:
    rev=N*price*12
    cogs=N*12*(inf+support+1)+fixed_infra+rev*payfee
    gp=rev-cogs
    cacspend=N*churn*12*cac
    op=gp-cacspend-other_opex
    print(f"N={N}: revenue ${rev:,.0f} ({fmt(rev*FX)}) cogs ${cogs:,.0f} GP ${gp:,.0f} GM {gp/rev*100:.0f}% CAC/yr ${cacspend:,.0f} other ${other_opex:,.0f} Op pre-comp ${op:,.0f} ({fmt(op*FX)}) comp ${team:,.0f} after ${op-team:,.0f} ({fmt((op-team)*FX)}); one-time CAC ${N*cac:,.0f}")
print()
print("=== Enterprise private deployment, Rs ===")
acv=20e5; setup=5e5; infra=9e5; supp=4.3e5; cac=5e5
for N,team,other in [(1,0,3e5),(5,(2*1.2e5+1e5)*12,8e5),(20,(10*1.25e5+2e5)*12,40e5)]:
    rev=N*acv
    cogs=N*(infra+supp)
    gp=rev-cogs
    acq=N*0.15*cac  # 15% annual logo churn replacement
    op=gp-acq-other
    print(f"N={N}: recurring rev {fmt(rev)} (+setup yr1 {fmt(N*setup)}) cogs {fmt(cogs)} GP {fmt(gp)} GM {gp/rev*100:.0f}% acq {fmt(acq)} other {fmt(other)} Op pre-comp {fmt(op)} team {fmt(team)} after {fmt(op-team)}; one-time CAC {fmt(N*cac)}")
print()
print("=== Productized AI automation services (global SMB) ===")
for projects,price,maint_clients in [(6,2500,3),(12,3000,8),(24,3500,20)]:
    rev=projects*price+maint_clients*300*12
    tools=projects*150+maint_clients*30*12
    contractors=0 if projects<=12 else projects*price*0.35
    gp=rev-tools-contractors-rev*0.05
    print(f"{projects} projects/yr @${price} + {maint_clients} maint: revenue ${rev:,.0f} ({fmt(rev*FX)}) GP ${gp:,.0f} GM {gp/rev*100:.0f}%")
print()
# break-even India SMB: fixed = other opex + fixed infra + founder 50k/month
inf_c = 2000*inf_doc_usd*FX
contrib = 1500*(1-0.0236) - (inf_c+250+50) - 0.04*6000
print("India SMB contribution/cust/month Rs", round(contrib))
fixed_month = (6e5+15000*12+0.5e5*12)/12
print("Break-even customers (founder Rs50k/mo, opex Rs6L/yr, infra Rs15k/mo):", round(fixed_month/contrib))
contrib_g = 49*(1-0.05) - (14+3+1) - 0.03*150
fixed_g = (6e5/FX + 600*12 + 0.5e5*12/FX)/12
print("Global devtool contribution/team/month $", round(contrib_g,2), "break-even teams:", round(fixed_g/contrib_g))
