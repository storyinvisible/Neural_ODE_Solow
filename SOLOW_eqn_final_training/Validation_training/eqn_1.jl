include("training_methods.jl")
global i =2
columns= names(df)
eqn1_para=DataFrame()
eqn2_para=DataFrame()
eqn3_para=DataFrame()
while i < size(df)[2]
    country = split(columns[i],"_")[1:end-1][1]
    country_data=getCleandataFrame(country)
    years_available=size(country_data)[1]
    if years_available>20
        try
            parameters,pred_value, losses= train_eqn1_mulitplication(country,20, 5)
            parameters2,pred_value2, losses2= train_eqn2_mulitplication(country,parameters,20, 5)
            parameters3,pred_value3, losses3= train_eqn3_mulitplication(country,parameters2,parameters2,20, 5)
            eqn1_para[country]= parameters
            eqn2_para[country]= parameters2
            eqn3_para[country]= parameters3
        catch e
            if isa(e, InterruptException)
                break
            end
        end


    end

    global i=i+7
end
CSV.write("Eqn1.csv", eqn1_para)
CSV.write("Eqn2.csv", eqn2_para)
CSV.write("Eqn3.csv", eqn3_para)
