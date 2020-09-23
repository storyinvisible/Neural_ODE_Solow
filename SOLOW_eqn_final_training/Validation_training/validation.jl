include("..\\multiplication\\Experiment_3_methods.jl")
cd(@__DIR__)
using LaTeXStrings
training_1=CSV.read("Eqn1.csv")
training_2=CSV.read("Eqn2.csv")
training_3=CSV.read("Eqn3.csv")

function pred(country, nnp1, nnp2,nnp3,test_year)
        newdf=getCleandataFrame(country)
        years_available=size(newdf)[1]
        st = interpolate(newdf["srate"], BSpline(Quadratic(Reflect(OnCell()))));
                nt = interpolate(newdf["nt"], BSpline(Quadratic(Reflect(OnCell()))));
                et = interpolate(newdf["et"], BSpline(Quadratic(Reflect(OnCell()))));
                kt = interpolate(newdf["k"], BSpline(Quadratic(Reflect(OnCell()))));
                yt = interpolate(newdf["yt"], BSpline(Quadratic(Reflect(OnCell()))));
                p1 =Parameter_parsin!(newdf, country)
                kdata=newdf["k"]
                ydata= newdf["yt"]
                edata=newdf["et"]
                dedt=newdf["dedt"]
                dydt=newdf["dydt"]
                kic = kdata[years_available-test_year]
                yic = ydata[years_available-test_year]
                # # eₜ = et(t)
                eic =  edata[years_available-test_year]
                ann = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                        FastDense(20, 1))
                ann2 = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                        FastDense(20, 1))
                ann3 = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                        FastDense(20, 1))
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
                function solownneqn1(du,u,nnp,t)
                    sₜ = st(t) #savings rate
                    nₜ = nt(t)
                    yₜ = yt(t)
                    kₜ = kt(t)
                    α,θ,δ,gᵦ = p1
                    k = u[1]
                    y=u[2]
                    e=u[3]
                    nn = ann(t,nnp1)[1]
                    nn2= ann2(t,nnp2)[1]
                    nn3= ann3(t,nnp3)[1]
                    du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k*nn # rate of change of kt
                    du[2] = y*(gᵦ+α*du[1]/kₜ)*nn2 # rate of change of yt
                    du[3] = e*(-gₐ+du[2]/y) *nn3

                    return du
                end
                nntspan = (years_available-test_year*1.0,years_available*1.0) ## the year is hard coded, to be changed later
                nndatasize = test_year
                nntsteps = range(nntspan[1], nntspan[2], length=nndatasize)
                print(nntsteps)
                u0 = Float32[kic,yic,eic] ## kt initial

                prob_solownneqn = ODEProblem(solownneqn1, u0, nntspan, 0)
                nnsolution = solve(prob_solownneqn, Tsit5(),p=0, abstol=1e-8, reltol=1e-8, saveat = nntsteps)
                Pred_plot(yt,et,kt,nnsolution)
        end
function Pred_plot(yt,et,kt,solution)
    fpl= plot([],label=" ",color=:white,legendfontsize=1)
    scatter!(steps,solution[1,:],label=L"k{sol}", color=:red, legend=(0.20,1.0),
    left_margin = 15Plots.mm,foreground_color_legend = nothing,background_color_legend = nothing,ylabel=L"k^c_t,y^t_c" ,
    markershape=:circle)
    scatter!(fpl,steps,solution[2,:],label=L"y_{sol}",color=:blue,markershape=:diamond)
    plot!(kt,kt,label=L"k_{data}",color=:red)
    plot!(fpl,yt,label=L"y_{data}",color=:blue,linestyle=:dash)
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
for country in names(training_3)
        pred(country, training_1[country], training_2[country],training_3[country],5)
end
