#=
This file is to simulate the new experiment of the solow equation with deep learning
With less neural nodes, taken in T as parameters, and multiplications of neural netwwork
instead of the summation
=#

cd(@__DIR__)


using OrdinaryDiffEq
using ModelingToolkit
using DataDrivenDiffEq
using LinearAlgebra, DiffEqSensitivity, Optim
using DiffEqFlux, Flux
using Plots, StatsPlots
using DataFrames, GLM, StatsModels
using CSV
using Interpolations
using Statistics
gr()
df = CSV.read("compileddf.csv",normalizenames = true)
column_names=names(df)
column_names[2:end]
##
eqn_1_para= Dict()
nnp_parameters= DataFrame()
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
    return[α, θ,δ, gᵦ]
end
function trainiter!(res1, rate,loss,callback)
    res1= DiffEqFlux.sciml_train(loss, res1.minimizer, ADAM(rate),cb=callback, maxiters=300)
    res1= DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=rate/100),cb=callback, maxiters=300)
    res1= DiffEqFlux.sciml_train(loss, res1.minimizer, ADAM(rate/10),cb=callback, maxiters=300)
    res1= DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=rate/100),cb=callback, maxiters=300)
    return res1
end

function ToTrainAndPlot_eqn1(prob_solownneqn, nntsteps,nnp,u0,actual,country,ann,control)
    current_parameters=nnp
    nnsolution = solve(prob_solownneqn, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)
    nnsolution[1,:]
    function predict(θ)
         return Array(solve(prob_solownneqn, Tsit5(),u0 = u0, p=θ, saveat = nntsteps))
    end
    function loss(θ)
        pred = predict(θ)

        return Flux.mse(actual, pred[1,:]), pred
    end
    losses = []
    callback(θ,l,pred) = begin
        push!(losses, l)
        nnpred = predict(θ)
        plknn = scatter(nntsteps,nnpred[1,:],label="predicted",legendfontsize=5)
        plot!(plknn,actual,label="true_k")
        plot!(legend=:bottomright)
        loss_plot=plot(losses, yaxis=:log)
        nn = plot(ann(nntsteps', θ)', title="Network")
        display(plot(plknn,loss_plot,nn,layout=3))
        if length(losses)%99==0 #print every 50 steps
            println(losses[end])

        end
        false
    end

    res1= DiffEqFlux.sciml_train(loss,nnp, ADAM(0.001),cb=callback, maxiters=10)

    res1= DiffEqFlux.sciml_train(loss,nnp, ADAM(0.001),cb=callback, maxiters=300)
    try
        res2= DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01/100),cb=callback, maxiters=300)
        res1=res2
    catch e
        print(" error occured")
    end
    #

    # println("ADAM")
    res1= DiffEqFlux.sciml_train(loss,res1.minimizer, ADAM(0.0001),cb=callback, maxiters=300)
    try
        res2= DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01/100),cb=callback, maxiters=300)
        res1=res2
    catch e
        print("error occured")
    end
    nnpred = predict(res1.minimizer)

    return res1.minimizer,nnpred[1,:],losses
end

function ToTrainAndPlot_eqn2(prob_solownneqn, nntsteps,nnp,u0,actual,country,ann,control)
    current_parameters=nnp
    # nnsolution = solve(prob_solownneqn, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)
    # nnsolution[1,:]
    function predict(θ)
         return Array(solve(prob_solownneqn, Tsit5(),u0 = u0, p=θ, saveat = nntsteps))
    end
    function loss(θ)
        pred = predict(θ)

        return Flux.mse(actual, pred[2,:]), pred
    end
    losses = []
    callback(θ,l,pred) = begin
        push!(losses, l)
        nnpred = predict(θ)
        plknn = scatter(nntsteps,nnpred[2,:],label="predicted",legendfontsize=5)
        plot!(plknn,actual,label="true_k")
        plot!(legend=:bottomright)
        loss_plot=plot(losses, yaxis=:log)
        nn = plot(ann(nntsteps', θ)', title="Network")
        display(plot(plknn,loss_plot,nn,layout=3))
        if length(losses)%99==0 #print every 50 steps
            println(losses[end])

        end
        false
    end

    res1= DiffEqFlux.sciml_train(loss,nnp, ADAM(0.001),cb=callback, maxiters=10)
    rate=0.01
    i=1
    while i <2
        try
            res1= trainiter!(res1, rate,loss,callback)
            rate= rate/100
        catch e
            if isa(e, InterruptException)
                break
            else
                println(string(country," had experience a problem"))
            end
        end
        i=i+1

    end

    nnpred = predict(res1.minimizer)
    # plknn = scatter(nntsteps,nnpred[2,:],label="preidcted_y",legendfontsize=5)
    # nnpred = predict(res1.minimizer)
    # plknn = scatter(nntsteps,nnpred[2,:],label="predicted",legend=:bottomright)
    # plot!(plknn,actual,label="true_y",title="Predicted vs Actual ")
    # loss_plot=plot(losses, title="Loss over iteration",yaxis=:log ,ylabel="Iterations", xlabel="Loss",label=NaN)
    # nn = plot(ann(nntsteps', res1.minimizer)',label=nothing,title= "Neural Network")
    # control_plot= scatter(nntsteps,control, label="y by solver" , title="Acutal vs Solver",legend=:topleft,legendfontsize=5)
    # plot!(control_plot,actual, label="true k")
    # plot(plknn,loss_plot,nn,control_plot,layout=4)
    # filename= string(country, "_eqn_2")
    # savefig(string("./performance/",filename))
    return res1.minimizer,nnpred[2,:],losses
end
function ToTrainAndPlot_eqn3(prob_solownneqn, nntsteps,nnp,u0,actual,country,ann,control)
    current_parameters=nnp
    # nnsolution = solve(prob_solownneqn, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)
    # nnsolution[1,:]
    function predict(θ)
         return Array(solve(prob_solownneqn, Tsit5(),u0 = u0, p=θ, saveat = nntsteps))
    end
    function loss(θ)
        pred = predict(θ)

        return Flux.mse(actual, pred[3,:]), pred
    end
    losses = []
    callback(θ,l,pred) = begin
        push!(losses, l)
        nnpred = predict(θ)
        plknn = scatter(nntsteps,nnpred[3,:],label="predicted e",legendfontsize=5)
        plot!(plknn,actual,label="true_e")
        plot!(legend=:bottomright)
        loss_plot=plot(losses, yaxis=:log)
        nn = plot(ann(nntsteps', θ)', title="Network")
        display(plot(plknn,loss_plot,nn,layout=3))
        if length(losses)%99==0 #print every 50 steps
            println(losses[end])

        end
        false
    end

    res1= DiffEqFlux.sciml_train(loss,nnp, ADAM(0.001),cb=callback, maxiters=10)
    rate=0.01
    i=1
    while i <2
        try
            res1= trainiter!(res1, rate,loss,callback)
            rate= rate/100
        catch e
            if isa(e, InterruptException)
                break
            else
                println(string(country," had experience a problem"))
            end
        end
        i=i+1

    end

    nnpred = predict(res1.minimizer)
    # plknn = scatter(nntsteps,nnpred[3,:],label="preidcted_e",legendfontsize=5)
    # nnpred = predict(res1.minimizer)
    # plknn = scatter(nntsteps,nnpred[3,:],label="predicted",legend=:bottomright)
    # plot!(plknn,actual,label="true_e",title="Predicted vs Actual ")
    # loss_plot=plot(losses, title="Loss over iteration",yaxis=:log ,ylabel="Iterations", xlabel="Loss",label=NaN)
    # nn = plot(ann(nntsteps', res1.minimizer)',label=nothing,title= "Neural Network")
    # control_plot= scatter(nntsteps,control, label="k by solver" , title="Acutal vs Solver",legend=:topleft,legendfontsize=5)
    # plot!(control_plot,actual, label="true e")
    # plot(plknn,loss_plot,nn,control_plot,layout=4)
    # filename= string(country, "_eqn_3")
    # savefig(string("./performance/",filename))
    return res1.minimizer,nnpred[3,:],losses
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
function train_eqn1_mulitplication(country,minimum_year, test_year)


        newdf=getCleandataFrame(country)
        years_available=size(newdf)[1]

        if(years_available>=minimum_year)
            st = interpolate(newdf["srate"], BSpline(Quadratic(Reflect(OnCell()))));
            nt = interpolate(newdf["nt"], BSpline(Quadratic(Reflect(OnCell()))));
            et = interpolate(newdf["et"], BSpline(Quadratic(Reflect(OnCell()))));
            kt = interpolate(newdf["k"], BSpline(Quadratic(Reflect(OnCell()))));
            yt = interpolate(newdf["yt"], BSpline(Quadratic(Reflect(OnCell()))));
            p1 =Parameter_parsin!(newdf, country)
            kdata=newdf["k"][1:end-test_year]
            ydata= newdf["yt"][1:end-test_year]
            kic = kt(1)
            yic = yt(1)
            # # eₜ = et(t)
            eic = et(1)
            ann = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                    FastDense(20, 1))
            nnp = initial_params(ann)
            function solownneqn1(du,u,nnp,t)
                sₜ = st(t) #savings rate
                nₜ = nt(t)
                yₜ = yt(t)

                θ,δ,gᵦ = p1
                k = u[1]
                nn = ann(t,nnp)[1]
                du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k*nn # rate of change of kt
                # du[2] = y*(gᵦ+α*du[1]/k) # rate of change of yt
                # du[3] = e*(-gₐ+du[2]/y)

                return du
            end
            function solownneqn1_control(du,u,nnp,t)
                sₜ = st(t) #savings rate
                nₜ = nt(t)
                yₜ = yt(t)

                θ,δ,gᵦ = p1
                k = u[1]
                nn = ann(t,nnp)[1]
                du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k # rate of change of kt
                # du[2] = y*(gᵦ+α*du[1]/k) # rate of change of yt
                # du[3] = e*(-gₐ+du[2]/y)

                return du
            end
            show(country)


            nntspan = (1.0e0,years_available-test_year) ## the year is hard coded, to be changed later
            nndatasize = years_available-test_year
            nntsteps = range(nntspan[1], nntspan[2], length=nndatasize)
            u0 = Float32[kic] ## kt initial

            prob_solownneqn = ODEProblem(solownneqn1, u0, nntspan, nnp)
            prob_solownneqn_control = ODEProblem(solownneqn1_control, u0, nntspan, nnp)
            nncontrol = solve(prob_solownneqn_control, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)

            ## function function required for training


            parameters,pred_value, losses= ToTrainAndPlot_eqn1(prob_solownneqn, nntsteps,nnp,u0,kdata,country,ann,nncontrol[1,:])

            return parameters,pred_value, losses
            # show(country)
            # show(" The loss is final loss is ")
            # show(final_loss)
            # show(" \n")
            #
            # plot(losses, fmt = :png, title="Loss over iteration" ,ylabel="loss",xlabel="iteration")
            # savefig(string("./equation1/loss_",country))
            # nnpred = predict(res1.minimizer)
            # plknn = scatter(nntsteps,nnpred[1,:],label="k")
            # plot!(plknn,kₜ,label="true_k")
            # plot!(legend=:bottomright)
            # plot(plknn,xlabel = "t")
            # savefig(string("./equation1/k_",country))
            # Statistics[country]=res1.minimizer

        end

end

function train_eqn2_mulitplication(country, nnp1,minimum_year,test_year )


    newdf=getCleandataFrame(country)
    years_available=size(newdf)[1]

    if(years_available>=10)
        st = interpolate(newdf["srate"], BSpline(Quadratic(Reflect(OnCell()))));
        nt = interpolate(newdf["nt"], BSpline(Quadratic(Reflect(OnCell()))));
        et = interpolate(newdf["et"], BSpline(Quadratic(Reflect(OnCell()))));
        kt = interpolate(newdf["k"], BSpline(Quadratic(Reflect(OnCell()))));
        yt = interpolate(newdf["yt"], BSpline(Quadratic(Reflect(OnCell()))));
        p1 =Parameter_parsin!(newdf, country)
        kdata=newdf["k"][1:end-test_year]
        ydata= newdf["yt"][1:end-test_year]
        kic = kt(1)
        yic = yt(1)
        # # eₜ = et(t)
        eic = et(1)
        ann = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                FastDense(20, 1))
        ann2 = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                FastDense(20, 1))
        nnp = initial_params(ann2)
        function solownneqn1(du,u,nnp,t)
            sₜ = st(t) #savings rate
            nₜ = nt(t)
            yₜ = yt(t)
            kₜ = kt(t)
            α,θ,δ,gᵦ = p1
            k = u[1]
            y=u[2]
            nn = ann(t,nnp1)[1]
            nn2= ann2(t,nnp)[1]
            du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k*nn # rate of change of kt
            du[2] = y*(gᵦ+α*du[1]/kₜ)*nn2 # rate of change of yt
            # du[3] = e*(-gₐ+du[2]/y)

            return du
        end
        function solownneqn1_control(du,u,nnp,t)
            sₜ = st(t) #savings rate
            nₜ = nt(t)
            yₜ = yt(t)

            α,θ,δ,gᵦ = p1
            k = u[1]
            y=u[2]
            du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k # rate of change of kt
            du[2] = y*(gᵦ+α*du[1]/k) # rate of change of yt
            # du[3] = e*(-gₐ+du[2]/y)

            return du
        end
        show(country)


        nntspan = (1.0e0,years_available-test_year) ## the year is hard coded, to be changed later
        nndatasize = years_available-test_year
        nntsteps = range(nntspan[1], nntspan[2], length=nndatasize)
        u0 = Float32[kic,yic] ## kt initial

        prob_solownneqn = ODEProblem(solownneqn1, u0, nntspan, nnp)
        nnsolution = solve(prob_solownneqn, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)

        prob_solownneqn_control = ODEProblem(solownneqn1_control, u0, nntspan, nnp)
        nncontrol = solve(prob_solownneqn_control, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)

        ## function function required for training


        para,pred_value, losses= ToTrainAndPlot_eqn2(prob_solownneqn, nntsteps,nnp,u0,ydata,country,ann,nncontrol[2,:])


        # show(country)
        # show(" The loss is final loss is ")
        # show(final_loss)
        # show(" \n")
        #
        # plot(losses, fmt = :png, title="Loss over iteration" ,ylabel="loss",xlabel="iteration")
        # savefig(string("./equation1/loss_",country))
        # nnpred = predict(res1.minimizer)
        # plknn = scatter(nntsteps,nnpred[1,:],label="k")
        # plot!(plknn,kₜ,label="true_k")
        # plot!(legend=:bottomright)
        # plot(plknn,xlabel = "t")
        # savefig(string("./equation1/k_",country))
        # Statistics[country]=res1.minimizer

    end



return para,pred_value, losses
end


# plot!(china_predplot,china_actual,label="true_k")
#
# china_predplot=plot(china_predplot,xlabel = "t",title="pred k vs actual k")
# china_lossplot=plot(china_losses, yaxis=:log,title="Loss over iteration")
# plot(china_predplot,china_lossplot,layout=2)


function train_eqn3_mulitplication(country, nnp1,nnp2,minimum_year,test_year)


    newdf=getCleandataFrame(country)
    years_available=size(newdf)[1]

    if(years_available>=minimum_year)
        st = interpolate(newdf["srate"], BSpline(Quadratic(Reflect(OnCell()))));
        nt = interpolate(newdf["nt"], BSpline(Quadratic(Reflect(OnCell()))));
        et = interpolate(newdf["et"], BSpline(Quadratic(Reflect(OnCell()))));
        kt = interpolate(newdf["k"], BSpline(Quadratic(Reflect(OnCell()))));
        yt = interpolate(newdf["yt"], BSpline(Quadratic(Reflect(OnCell()))));
        p1 =Parameter_parsin!(newdf, country)
        kdata=newdf["k"][1:end-test_year]
        ydata= newdf["yt"][1:end-test_year]
        edata=newdf["et"][1:end-test_year]
        dedt=newdf["dedt"][1:end-test_year]
        dydt=newdf["dydt"][1:end-test_year]
        kic = kt(1)
        yic = yt(1)
        # # eₜ = et(t)
        eic = et(1)
        ann = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                FastDense(20, 1))
        ann2 = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                FastDense(20, 1))
        ann3 = FastChain(FastDense(1, 20, tanh),FastDense(20, 20, tanh),
                FastDense(20, 1))
        nnp = initial_params(ann3)
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
            nn3= ann3(t,nnp)[1]
            du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k*nn # rate of change of kt
            du[2] = y*(gᵦ+α*du[1]/kₜ)*nn2 # rate of change of yt
            du[3] = e*(-gₐ+du[2]/y) *nn3

            return du
        end
        function solownneqn1_control(du,u,nnp,t)
            sₜ = st(t) #savings rate
            nₜ = nt(t)
            yₜ = yt(t)

            α,θ,δ,gᵦ = p1
            k = u[1]
            y=u[2]
            e=u[3]
            du[1] = sₜ*yₜ-(δ+nₜ+gᵦ)*k # rate of change of kt
            du[2] = y*(gᵦ+α*du[1]/k) # rate of change of yt
            du[3] = e*(-gₐ+du[2]/y)

            return du
        end
        show(country)


        nntspan = (1.0e0,years_available-test_year) ## the year is hard coded, to be changed later
        nndatasize = years_available-test_year
        nntsteps = range(nntspan[1], nntspan[2], length=nndatasize)
        u0 = Float32[kic,yic,eic] ## kt initial

        prob_solownneqn = ODEProblem(solownneqn1, u0, nntspan, nnp)
        nnsolution = solve(prob_solownneqn, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)

        prob_solownneqn_control = ODEProblem(solownneqn1_control, u0, nntspan, nnp)
        nncontrol = solve(prob_solownneqn_control, Tsit5(),p=nnp, abstol=1e-8, reltol=1e-8, saveat = nntsteps)

        ## function function required for training


        para,pred_value, losses= ToTrainAndPlot_eqn3(prob_solownneqn, nntsteps,nnp,u0,edata,country,ann3,nncontrol[3,:])


        # show(country)
        # show(" The loss is final loss is ")
        # show(final_loss)
        # show(" \n")
        #
        # plot(losses, fmt = :png, title="Loss over iteration" ,ylabel="loss",xlabel="iteration")
        # savefig(string("./equation1/loss_",country))
        # nnpred = predict(res1.minimizer)
        # plknn = scatter(nntsteps,nnpred[1,:],label="k")
        # plot!(plknn,kₜ,label="true_k")
        # plot!(legend=:bottomright)
        # plot(plknn,xlabel = "t")
        # savefig(string("./equation1/k_",country))
        # Statistics[country]=res1.minimizer

    end



    return para,pred_value, losses
end
