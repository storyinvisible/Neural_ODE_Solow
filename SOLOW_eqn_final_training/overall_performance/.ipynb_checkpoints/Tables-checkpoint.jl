cd(@__DIR__)
using DataFrames
using CSV

eqn1_multi= CSV.read("..\\multiplication\\Eqn1.csv",normalizenames = true)
