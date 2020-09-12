cd(@__DIR__)
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
eqn1_multi= CSV.read(".\\multiplication\\Eqn1.csv",normalizenames = true)
df = CSV.read(".\\multiplication\\compileddf.csv",normalizenames = true)

column_names=names(df)
column_names[2:end]
##

mutable struct Frame
           dataframe
       end
mutable struct Frame2
    dataframe
end
frame = Frame(df)
frame_dk= Frame(df)
theta=[]
function eqn_1(α,θ)

# nn = ann(α,nnp)
    income = (1-θ).* frame.dataframe["k"].^ (α)
   return income
end
function exp_2_eqn2_mse(δ,gᵦ)
    len= length(theta)
    θ = theta[len]
     #fraction of economic activity dedicated to abatement
    dkdt = frame.dataframe["srate"] .* (1-θ) .* frame.dataframe["yt"] .- (δ .+ frame.dataframe["nt"] .+ gᵦ) .* frame.dataframe["k"] # rate of change of kt
    return dkdt
end
function exp2_eqn1_loss_mse(x)
    α = x[1]
    θ = x[2]
    pred= eqn_1(α,θ)
#     return sum(abs2, (yₜ .- pred))
#     return Flux.mse(yₜ[1:15,:], pred[1:15,:])
    return Flux.mse(frame.dataframe["yt"], pred)
end
function exp2_eqn2_loss_mse(y)
    δ = y[1]
    gᵦ = y[2]
    pred= exp_2_eqn2_mse(δ,gᵦ)

#     return sum(abs2, (yₜ .- pred))
#     return Flux.mse(yₜ[1:15,:], pred[1:15,:])
    return Flux.mse(frame_dk.dataframe, pred)
end
function Parameter_parsin!(cdf, country)

    frame_dk.dataframe= cdf["dkdt"]
    frame.dataframe = cdf
    dkdt_df=DataFrame()
    actual_dkdt_cn=string(country,"_actual_dkdt")
    #Experiment 2 mse
    optimizer1 = optimize(exp2_eqn1_loss_mse, [0.0,0.0])
    α,θ = Optim.minimizer(optimizer1)
    push!(theta, θ)
    optmizer2=optimize(exp2_eqn2_loss_mse, [0.0,0.0])
    δ, gᵦ=Optim.minimizer(optmizer2)
    edata= cdf["et"]
    ydata=cdf["yt"]
    dydt=cdf["dydt"]
    dedt=cdf["dedt"]
    function soloweqn3alt(gₐ)
#     dy = yₜ[2:end] .* (gᵦ) .+ α .* dk ./ kₜ[2:end] # rate of change of yt
    de = edata .* (-gₐ .+ dydt ./ ydata)
    end
    function losseqn3(xx)
        gₐ = xx
        pred= soloweqn3alt(gₐ)
        return Flux.mse(dedt, pred)
    end
    reseqn3 = optimize(losseqn3, -200.0, 200.0)
    gₐ = Optim.minimizer(reseqn3)
    return[α, θ,δ, gᵦ,gₐ]
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
paras= DataFrame()
global i =1
while i < length(column_names)
    country_string=column_names[i+1]
    country=join(split(country_string,"_")[1:end-1],"_")
    cdf=getCleandataFrame(country)
    years_available=size(cdf)[1]
    if(years_available>=10)
        para_individual=Parameter_parsin!(cdf, country)
        paras[country]= para_individual
    end
    global i = i +7
end
CSV.write("para.csv", paras)
