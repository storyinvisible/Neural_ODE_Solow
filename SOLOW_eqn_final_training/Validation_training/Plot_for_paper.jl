
include("..\\multiplication\\Experiment_3_methods.jl")
using LaTeXStrings
Aus_acutal= getCleandataFrame("Australia")
para_1,pred_value_1, losses_1=train_eqn1_mulitplication("Australia")
para_2,pred_value_2, losses_2=train_eqn2_mulitplication("Australia",para_1)
para_3,pred_value_3, losses_3=train_eqn3_mulitplication("Australia",para_1,para_2)
CSV.write("para_1_aus.csv",[para_1,para_2,para_3])
##
span = (1.0, size(Aus_acutal)[1])
steps= range(span[1], span[2], length=size(Aus_acutal)[1])
pl = scatter(steps,pred_value_1,label="k_{sol}", color=:red, legend=:topleft,ylabel=L"k^c_t,y^t_c",
    left_margin = 15Plots.mm,foreground_color_legend = nothing,background_color_legend = nothing,
    markershape=:circle)
scatter!(pl,steps,pred_value_2,label=L"y_{sol}",color=:blue,markershape=:diamond)
pr = twinx()
scatter!(pr,steps,pred_value_3,label=L"e_{sol}",color=:green, legend=:topleft,
    ylabel=L"e^t_c",right_margin = 15Plots.mm,markershape=:xcross)
plot!(pl,Aus_acutal["k"],label=L"k_{data}",color=:red)
plot!(pl,Aus_acutal["yt"],label=L"y_{data}",color=:blue,linestyle=:dash)
plot!(pr,Aus_acutal["et"],label=L"e_{data}",color=:green,framestyle = :box,linestyle=:dot,
    foreground_color_legend = nothing,background_color_legend = nothing, ylim=(0,35))

display(plot(pl,xlabel = "t",dpi=300,legendfontsize = 12,tickfont=12, guidefont = 16))
savefig("model1USA.png")
##
