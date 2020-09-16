cd("D:\\NUS_differential_equation\\SOLOW_eqn_final_training\\")
using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots, StatsPlots
using DataFrames, GLM, StatsModels
using CSV
using Interpolations
gr()


eqn1_multi= CSV.read(".\\multiplication\\Eqn1.csv",normalizenames = true)
eqn1_multi_loss= CSV.read(".\\multiplication\\Eqn1_final_loss.csv",normalizenames = true)

eqn2_multi= CSV.read(".\\multiplication\\Eqn2.csv",normalizenames = true)
eqn2_multi_loss= CSV.read(".\\multiplication\\Eqn2_final_loss.csv",normalizenames = true)

eqn3_multi= CSV.read(".\\multiplication\\Eqn3.csv",normalizenames = true)
eqn3_multi_loss= CSV.read(".\\multiplication\\Eqn3_final_loss.csv",normalizenames = true)

eqn1_add= CSV.read(".\\addition\\Eqn1.csv",normalizenames = true)
eqn1_add_loss= CSV.read(".\\addition\\Eqn1_final_loss.csv",normalizenames = true)

eqn2_add= CSV.read(".\\addition\\Eqn2.csv",normalizenames = true)
eqn2_add_loss= CSV.read(".\\addition\\Eqn2_final_loss.csv",normalizenames = true)

eqn3_add= CSV.read(".\\addition\\Eqn3.csv",normalizenames = true)
eqn3_add_loss= CSV.read(".\\addition\\Eqn3_final_loss.csv",normalizenames = true)
df= CSV.read(".\\multiplication\\compileddf.csv",normalizenames = true)
paras= CSV.read("para.csv")

ann = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                FastDense(20, 1))

function solownneqn(du,u,p,t)
    country= p
    newdf= getCleandataFrame(country)
    st = interpolate(newdf["srate"], BSpline(Quadratic(Reflect(OnCell()))));
    nt = interpolate(newdf["nt"], BSpline(Quadratic(Reflect(OnCell()))));
    et = interpolate(newdf["et"], BSpline(Quadratic(Reflect(OnCell()))));
    kt = interpolate(newdf["k"], BSpline(Quadratic(Reflect(OnCell()))));
    yt = interpolate(newdf["yt"], BSpline(Quadratic(Reflect(OnCell()))));
    sₜ = st(t) #savings rate
    nₜ = nt(t)
    yₜ = yt(t)
    kₜ = kt(t)
    nnp1= eqn1_multi[country]
    nnp2= eqn2_multi[country]
    nnp3= eqn3_multi[country]
    α,θ,δ,gᵦ,gₐ = paras[country]
    k = u[1]
    y=u[2]
    e=u[3]
    nn = ann(t,nnp1)[1]
    nn2= ann(t,nnp2)[1]
    nn3= ann(t,nnp3)[1]
    du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k*nn # rate of change of kt
    du[2] = y*(gᵦ+α*du[1]/kₜ)*nn2 # rate of change of yt
    du[3] = e*(-gₐ+du[2]/y) *nn3

    return du
end


function solownneqn_add(du,u,p,t)
    country= p
    newdf= getCleandataFrame(country)
    st = interpolate(newdf["srate"], BSpline(Quadratic(Reflect(OnCell()))));
    nt = interpolate(newdf["nt"], BSpline(Quadratic(Reflect(OnCell()))));
    et = interpolate(newdf["et"], BSpline(Quadratic(Reflect(OnCell()))));
    kt = interpolate(newdf["k"], BSpline(Quadratic(Reflect(OnCell()))));
    yt = interpolate(newdf["yt"], BSpline(Quadratic(Reflect(OnCell()))));
    sₜ = st(t) #savings rate
    nₜ = nt(t)
    yₜ = yt(t)
    kₜ = kt(t)
    nnp1= eqn1_multi[country]
    nnp2= eqn2_multi[country]
    nnp3= eqn3_multi[country]
    α,θ,δ,gᵦ,gₐ = paras[country]
    k = u[1]
    y=u[2]
    e=u[3]
    nn = ann(t,nnp1)[1]
    nn2= ann(t,nnp2)[1]
    nn3= ann(t,nnp3)[1]
    du[1] = sₜ*y-(δ+nₜ+gᵦ)*k+nn # rate of change of kt
    du[2] = y*(gᵦ+α*du[1]/k)+nn2 # rate of change of yt
    du[3] = e*(-gₐ+du[2]/y) +nn3

    return du
end

function solownneqn1_control(du,u,p,t)
    country= p
    newdf= getCleandataFrame(country)
    st = interpolate(newdf["srate"], BSpline(Quadratic(Reflect(OnCell()))));
    nt = interpolate(newdf["nt"], BSpline(Quadratic(Reflect(OnCell()))));
    et = interpolate(newdf["et"], BSpline(Quadratic(Reflect(OnCell()))));
    kt = interpolate(newdf["k"], BSpline(Quadratic(Reflect(OnCell()))));
    yt = interpolate(newdf["yt"], BSpline(Quadratic(Reflect(OnCell()))));
    sₜ = st(t) #savings rate
    nₜ = nt(t)
    yₜ = yt(t)
    kₜ = kt(t)
    α,θ,δ,gᵦ,gₐ = paras[country]
    k = u[1]
    y=u[2]
    e=u[3]
    du[1] = sₜ*y-(δ+nₜ+gᵦ)*k # rate of change of kt
    du[2] = y*(gᵦ+α*du[1]/k) # rate of change of yt
    du[3] = e*(-gₐ+du[2]/y)

    return du
end
function getCleandataFrame(country)
    df_year=df["Year"][2:end]
    df1=df[string(country,"_yt")]  #GDP
    df2=df[string(country,"_kt")] #capital per capita
    df3=df[string(country,"_srate")][2:end] # savig rate
    df6= df[string(country,"_et")] #population
    df4= df[string(country,"_nt")][2:end] #population
    df8=df6[1:end-1] .-df6[2:end]#de
    df6=df6[2:end]
    df5=df2[1:end-1] .-df2[2:end] #dk
    df2=df2[2:end]
    df7= df1[1:end-1] .-df1[2:end] # dydt
    df1=df1[2:end]
    newdf= DataFrame(Year=df_year,yt=df1, k=df2, srate=df3,nt=df4,dkdt=df5,dydt=df7,dedt=df8,et=df6)
    newdf=newdf[completecases(newdf), :]
    return newdf
end
loss_summary_multi=DataFrame()
loss_summary_multi["header"]=["K_loss_multi","Y_loss_multi","E_loss_multi","K_control","Y_control","E_control"]
eqn3_multi = select(eqn3_multi, Not(:El_Salvador))
eqn3_multi = select(eqn3_multi,Not(:Jordan))
eqn3_multi = select(eqn3_multi, Not(:Morocco))
eqn3_multi = select(eqn3_multi, Not(:Nepal))
eqn3_multi = select(eqn3_multi, Not(:Philippines))
for country in names(eqn3_multi)
    print(country)
    newdf= getCleandataFrame(country)
    kic = newdf["k"][1]
    yic = newdf["yt"][1]
    eic = newdf["et"][1]
    years_available=size(newdf)[1]
    nntspan = (1.0e0,years_available)
    nndatasize = years_available
    nntsteps = range(nntspan[1], nntspan[2], length=nndatasize)
    u0 = Float32[kic,yic,eic]
    prob_solownneqn = ODEProblem(solownneqn, u0, nntspan, country)
    prob_solownneqn_control = ODEProblem(solownneqn1_control, u0, nntspan, country)
#     nnexperiment = solve(prob_solownneqn, Tsit5(),p=country, abstol=1e-8, reltol=1e-8, saveat = nntsteps)
    nncontrol = solve(prob_solownneqn_control, Tsit5(),p=country, abstol=1e-8, reltol=1e-8, saveat = nntsteps)

    k_loss=eqn1_multi_loss[country]
    k_control= Flux.mse(nncontrol[1,:],newdf["dkdt"])

    y_loss=eqn2_multi_loss[country]
    y_control= Flux.mse(nncontrol[2,:],newdf["dydt"])

    e_loss= eqn3_multi_loss[country]
    e_control= Flux.mse(nncontrol[3,:],newdf["dedt"])

    loss_summary_multi[country]= [k_loss[1],k_control,y_loss[1],y_control,e_loss[1],e_control]

end
loss_summary_add=DataFrame()
loss_summary_add["header"]=["K_loss_add","K_control","Y_loss_add","Y_control","E_loss_add","E_control"]
eqn3_add = select(eqn3_add, Not(:El_Salvador))
eqn3_add = select(eqn3_add,Not(:Jordan))
eqn3_add = select(eqn3_add, Not(:Morocco))
eqn3_add = select(eqn3_add, Not(:Nepal))
eqn3_add = select(eqn3_add, Not(:Philippines))
for country in names(eqn3_add)
    print(country)
    newdf= getCleandataFrame(country)
    kic = newdf["k"][1]
    yic = newdf["yt"][1]
    eic = newdf["et"][1]
    years_available=size(newdf)[1]
    nntspan = (1.0e0,years_available)
    nndatasize = years_available
    nntsteps = range(nntspan[1], nntspan[2], length=nndatasize)
    u0 = Float32[kic,yic,eic]
    prob_solownneqn = ODEProblem(solownneqn, u0, nntspan, country)
    prob_solownneqn_control = ODEProblem(solownneqn1_control, u0, nntspan, country)
#     nnexperiment = solve(prob_solownneqn, Tsit5(),p=country, abstol=1e-8, reltol=1e-8, saveat = nntsteps)
    nncontrol = solve(prob_solownneqn_control, Tsit5(),p=country, abstol=1e-8, reltol=1e-8, saveat = nntsteps)

    k_loss_add=eqn1_add_loss[country]
    k_control= Flux.mse(nncontrol[1,:],newdf["dkdt"])

     y_loss_add=eqn2_add_loss[country]
    y_control= Flux.mse(nncontrol[2,:],newdf["dydt"])

    e_loss= eqn3_add_loss[country]
    e_control= Flux.mse(nncontrol[3,:],newdf["dedt"])

#     try

#     y_loss_add=eqn2_add_loss[country]

    loss_summary_add[country]= [k_loss_add[1],k_control,y_loss_add[1], y_control,e_loss[1],e_control]
#         y_loss_add,y_control]
#                     e_loss_add,e_control

#     catch e
#         k_loss_add=missing
#         y_loss_add=missing
#         e_loss_add=missing
#         end



end
CSV.write(".//overall_performance//add_loss.csv", loss_summary_add)
CSV.write(".//overall_performance//multi_loss.csv", loss_summary_multi)
