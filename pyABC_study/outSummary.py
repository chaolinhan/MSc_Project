import pandas

out_data = pandas.read_csv(r'/home/yuan/wdnmd/MSc_Project/pyABC_study/outRaw.csv')

out_data = out_data.drop('id', axis=1)

paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

with open(r'/home/yuan/wdnmd/MSc_Project/pyABC_study/outSummary.csv', "w+") as out_file:

    for key in out_data:
        out_file.write(str(key) + ", mean, " + str(out_data[key].mean()) + "\n")

    for key in out_data:
        out_file.write(str(key) + ", sd, " + str(out_data[key].std()) + "\n")

    for key in paraInit:
        out_file.write(str(key) + ", initial, "+str(paraInit[key]) + "\n")