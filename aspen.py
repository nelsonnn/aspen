from sklearn import tree

training_feats = { "armada tracer 108" : [825, 180, 108, True, True, False],
         "armada arv 116 JJ" : [825, 192, 116, True, True, True],
         "armada tracer 118 CHX" : [950, 188, 118, True, True, False],
         "armada magic J" : [875, 190, 127, True, True, True],
         "armada bdog" : [650, 165, 90, False, True, False],
         "blizzard brahma" : [780, 180, 88, False, False, False],
         "blizzard bonafide" : [840, 166, 98, False, False, True],
         "rossignol experience 88 hd" : [1000, 172, 88, False, False, False],
         "rossignol sprayer xpress" : [450, 178, 80, False, True, True],
         "rossignol soul 7 hd" : [850, 188, 106, True, False, False],
         "rossignol sky 7 hd" : [750, 180, 98, True, False, True],
         "volkl racetiger gs r 30" : [900, 188, 65, False, False, False],
         "volkl rtm 86" : [1159, 167, 86, False, False, False],
         "volkl mantra" : [699, 191, 100, True, False, True],
         "volkl bash 116" : [649, 186, 116, True, True, True],
         "salomon s lab x alp" : [700, 164, 96, True, False, False],
         "salomon qst 118" : [749, 185, 118, True, False, False],
         "salomon qst 106" : [699, 181, 106, False, False, True],
         "salomon NFX" : [499, 176, 86, True, True, True],
         "icelantic nomad 105" : [749, 171, 105, True, True, True],
         }

testing_feats = { "icelantic pioneer 109" : [699, 190, 109, True, False, True],
         "icelantic vangaurd 97" : [699, 168, 97, False, False, True],
         "faction prime 3.0" : [1289, 189, 108, True, False, True],
         "faction prodigy 2.0" : [669, 174, 96, True, True, False],
         "black crows corvus" : [749, 188, 109, True, False, True],
         "black crows daemon" : [769, 177, 99, True, False, True]
         }

training_labels = { "armada tracer 108" : 1,
         "armada arv 116 JJ" : 1,
         "armada tracer 118 CHX" : 0,
         "armada magic J" : 1,
         "armada bdog" : 0,
         "blizzard brahma" : 0,
         "blizzard bonafide" : 0,
         "rossignol experience 88 hd" : 0,
         "rossignol sprayer xpress" : 0,
         "rossignol soul 7 hd" : 1,
         "rossignol sky 7 hd" : 1,
         "volkl racetiger gs r 30" : 0,
         "volkl rtm 86" : 0,
         "volkl mantra" : 1,
         "volkl bash 116" : 1,
         "salomon s lab x alp" : 0,
         "salomon qst 118" : 1,
         "salomon qst 106" : 1,
         "salomon NFX" : 0,
         "icelantic nomad 105" : 1
         }

testing_labels = { "icelantic pioneer 109" : 1,
         "icelantic vangaurd 97" : 1,
         "faction prime 3.0" : 0,
         "faction prodigy 2.0" : 1,
         "black crows corvus" : 1,
         "black crows daemon" : 1
         }

def main() :
    X = [ training_feats[key] for key in training_feats ]
    Y = [ training_labels[key] for key in training_labels ]

    skitree = tree.DecisionTreeClassifier()
    skitree.fit(X,Y)

main()
