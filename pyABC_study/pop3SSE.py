import pandas

pop3data = pandas.read_csv(r'/pyABC_study/pop3outRaw.csv')

pop3data = pop3data.drop('id', axis=1)

paraInit = {"lambdaN": 13.753031, "kNB": 1.581684, "muN": -0.155420, "vNM": 0.262360,
            "lambdaM": 2.993589, "kMB": 0.041040, "muM": 0.201963,
            "sBN": 1.553020, "iBM": -0.046259, "muB": 1.905163,
            "sAM": 11.001731, "muA": 23.022678}

for key in pop3data:
    print(str(key)+", sse, "+str(((pop3data[key] - paraInit[key]) ** 2).sum()))

for key in pop3data:
    print(str(key) + ", mean, "+str(pop3data[key].mean()))

for key in pop3data:
    print(str(key) + ", sd, "+str(pop3data[key].std()))

for key in paraInit:
    print(str(key) + ", initial, "+str(paraInit[key]))