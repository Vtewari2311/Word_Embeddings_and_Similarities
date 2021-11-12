import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
glove = api.load("glove-wiki-gigaword-300")
# testing functions
# print (wv ["king"])
# print (wv.similarity("king", "queen"))
# v = wv["king"] - wv["male"] + wv["female"]
# type(v)
# <class 'numpy.ndarray'>
# print(wv.similar_by_vector(v))
# testing evaluation of word pairs given a text file "sim.txt"
# tests on bigger file for word pair evaluation to get Pearson co-efficient using word_vec
# print(wv.evaluate_word_pairs("sim.txt"))
# tests on bigger file for word pair evaluation to get Pearson co-efficient using glove
# print(glove.evaluate_word_pairs("sim.txt"))
# testing evaluation of word analogies given text file "analogy.txt"
# a = wv.evaluate_word_analogies("analogy.txt")
# priting accuracy
# print (a[0])
# printing correctness
# print (a[1])


# TASK 1 : Evaluate word2vec and glove embeddings on the SimLexA2.txt dataset 
print(wv.evaluate_word_pairs("SimLexA2.txt"))
print(glove.evaluate_word_pairs("SimLexA2.txt"))

# Task 2 : 
# 1. Come up with at least 10 word pairs on your own and assign them similarity scores 0-1 based  on your judgement
# 2. Create your own tab-delimited file 
# 3. Record the similarity scores obtained using word2vec and glove for each word pair.

# Part 1 
# word_vec:
print(wv.similarity("dog", "cat"))
print(wv.similarity("football", "soccer"))
print(wv.similarity("cabinet", "seagull"))
print(wv.similarity("car", "bike"))
print(wv.similarity("internet", "cable"))
print(wv.similarity("key", "lock"))
print(wv.similarity("table", "top"))
print(wv.similarity("toilet", "bathroom"))
print(wv.similarity("latitude", "longitude"))
print(wv.similarity("picture", "frame"))
print(wv.similarity("lamp", "bulb"))
print(wv.similarity("outlet", "charger"))
print(wv.similarity("shoe", "lace"))

# glove:
print(glove.similarity("dog", "cat"))
print(glove.similarity("football", "soccer"))
print(glove.similarity("cabinet", "seagull"))
print(glove.similarity("car", "bike"))
print(glove.similarity("internet", "cable"))
print(glove.similarity("key", "lock"))
print(glove.similarity("table", "top"))
print(glove.similarity("toilet", "bathroom"))
print(glove.similarity("latitude", "longitude"))
print(glove.similarity("picture", "frame"))
print(glove.similarity("lamp", "bulb"))
print(glove.similarity("outlet", "charger"))
print(glove.similarity("shoe", "lace"))

# Part 3
print(wv.evaluate_word_pairs("qs2.txt"))
print(glove.evaluate_word_pairs("qs2.txt"))

# Negative correlation
#print(wv.similarity("cabinet", "seagull"))
# result = 0.009027985
# print(glove.similarity("cabinet", "seagull"))
# result = -0.07938038

# Task 3 : Come up with at least 10  word analogy questions on your own and write them in the format needed by Gensim. Evaluate both word2vec and glove on your dataset.

a = wv.evaluate_word_analogies("qs3.txt")
print(a[0])
print(a[1])
b = glove.evaluate_word_analogies("qs3.txt")
print(b[0])
print(b[1])

# Task 4 : Choose any 10 words and find their most similar five words using word2vec and glove embeddings.

# word_vec
print(wv.most_similar("raccoon"))
#[('raccoons', 0.7092068791389465), ('rabid_raccoon', 0.671484649181366), ('bobcat', 0.6711269617080688), ('squirrel', 0.6657680869102478), ('coyote', 0.6650583744049072), ('racoon', 0.6403231024742126), ('stray_cat', 0.6383534073829651), ('skunk', 0.6282143592834473), ('cat', 0.6210863590240479), ('critter', 0.6150866746902466)]
print(wv.most_similar("soldier"))
#[('solider', 0.9117935299873352), ('serviceman', 0.7837359309196472), ('soldiers', 0.7634838223457336), ('airman', 0.6886240839958191), ('guardsman', 0.6794973015785217), ('Soldier', 0.6792623400688171), ('Pfc', 0.6733466982841492), ('paratrooper', 0.663852334022522), ('soliders', 0.6594691872596741), ('corporal', 0.6549087762832642)]
print(wv.most_similar("guardian"))
#[('guardians', 0.7329328656196594), ('guardianship', 0.5965357422828674), ('played_Holger_Palmgren', 0.5946720242500305), ('Leslie_Andino', 0.5900617837905884), ('ad_Litem', 0.5346183776855469), ('litem', 0.5337220430374146), ('guardian_ad_litem', 0.5231743454933167), ('cousin_Montejo_Gaspar', 0.515192449092865), ('chaperon', 0.49579188227653503), ('executor', 0.4842901825904846)]
print(wv.most_similar("angel"))
#[('angels', 0.7340543270111084), ('luminous_cocoon', 0.554248571395874), ('guardian_angel', 0.5442337989807129), ('Seliethia_Parker', 0.5338423848152161), ('Clarence_Odbody', 0.5276748538017273), ('sure_housekeeper_Kaa', 0.5191166400909424), ('cherub', 0.5047515630722046), ('Angel', 0.5040815472602844), ('archangel', 0.5031775236129761), ('angel_investor', 0.5009891390800476)]
print(wv.most_similar("grass"))
#[('grasses', 0.6635476350784302), ('Bermuda_grass', 0.6476073265075684), ('bermuda_grass', 0.6298290491104126), ('rye_grass', 0.6293388605117798), ('lawns', 0.6126667261123657), ('fescue', 0.6100977063179016), ('Zoysia_grass', 0.6073623895645142), ('kikuyu', 0.6040306091308594), ('kikuyu_grass', 0.5956955552101135), ('tall_fescue_grass', 0.5796751976013184)]
print(wv.most_similar("laptop"))
#[('laptops', 0.805374026298523), ('laptop_computer', 0.7848465442657471), ('notebook', 0.67857825756073), ('netbook', 0.6707929372787476), ('computer', 0.6640493273735046), ('laptop_computers', 0.6633790731430054), ('notebook_PC', 0.6631842851638794), ('MacBook', 0.6598750352859497), ('PowerBook', 0.6520565748214722), ('Sony_Vaio_laptop', 0.6496156454086304)]
print(wv.most_similar("lamp"))
#[('lamps', 0.743071436882019), ('Rubbery_pizza_languishing', 0.5924029350280762), ('tealight', 0.5723157525062561), ('candle', 0.5694510340690613), ('lantern', 0.5688028931617737), ('bulb', 0.5665534734725952), ('candleholder', 0.5622106790542603), ('sconce', 0.56000816822052), ('fluorescent_bulb', 0.5478653311729431), ('wall_sconce', 0.5446075797080994)]
print(wv.most_similar("globe"))
#[('world', 0.6945998072624207), ('worldwide', 0.647681474685669), ('continents', 0.6263391971588135), ('continent', 0.6154399514198303), ('globally', 0.5908562541007996), ('theworld', 0.5461979508399963), ('priests_shuffled', 0.5357858538627625), ('world.The', 0.5146374106407166), ('country', 0.5137460827827454), ('global', 0.5118445754051208)]
print(wv.most_similar("bird"))
#[('birds', 0.8141971230506897), ('raptor', 0.6830927729606628), ('owl', 0.6825829148292542), ('squirrel', 0.6653631329536438), ('falcon', 0.6649249196052551), ('raptors', 0.6613034605979919), ('bald_eagle', 0.6493615508079529), ('robin', 0.6482654809951782), ('pelican', 0.6474649906158447), ('avian', 0.6436793804168701)]
print(wv.most_similar("bee"))
#[('bees', 0.7053181529045105), ('honeybee', 0.6075024604797363), ('spelling_bee', 0.5671892166137695), ('honey_bee', 0.5634711384773254), ('honey_bees', 0.5585135221481323), ('bumble_bees', 0.5508747696876526), ('honeybees', 0.5490716099739075), ('insect', 0.5376728177070618), ('bumble_bee', 0.5331851840019226), ("Jason_Giambi_jee_AHM'", 0.5291669964790344)]

# glove :
print(glove.most_similar("raccoon"))
#[('raccoons', 0.5740074515342712), ('squirrel', 0.5537731647491455), ('coyote', 0.5170983672142029), ('boar', 0.5077536702156067), ('mink', 0.5049654245376587), ('skunk', 0.4883301854133606), ('rabbit', 0.4861471951007843), ('muskrat', 0.48309701681137085), ('deer', 0.4758397936820984), ('bobcat', 0.47500160336494446)]
print(glove.most_similar("soldier"))
#[('soldiers', 0.7162267565727234), ('wounded', 0.6503106355667114), ('policeman', 0.6371234655380249), ('army', 0.5879426598548889), ('killed', 0.5516754984855652), ('serviceman', 0.5464208722114563), ('man', 0.5163717269897461), ('troops', 0.5148918628692627), ('policemen', 0.513936460018158), ('prisoner', 0.5128335356712341)]
print(glove.most_similar("guardian"))
#[('guardians', 0.4964340627193451), ('newspaper', 0.47482213377952576), ('reviewer', 0.42936545610427856), ('herald', 0.42828378081321716), ('editorial', 0.4262792766094208), ('tribune', 0.40509161353111267), ('commented', 0.4047023355960846), ('broadsheet', 0.4026525020599365), ('website', 0.3968515992164612), ('observer', 0.39581984281539917)]
print(glove.most_similar("angel"))
#[('miguel', 0.5479671359062195), ('jimenez', 0.5176248550415039), ('gabriel', 0.47134312987327576), ('jose', 0.47126585245132446), ('lopez', 0.470060795545578), ('gurria', 0.4505026340484619), ('garcia', 0.4437534511089325), ('ángel', 0.44344061613082886), ('angels', 0.4414026737213135), ('cabrera', 0.43476995825767517)]
print(glove.most_similar("grass"))
#[('grasses', 0.5605592727661133), ('lawn', 0.5430611968040466), ('pasture', 0.4857950210571289), ('lawns', 0.4798297584056854), ('roots', 0.4729629456996918), ('weeds', 0.4674966335296631), ('dirt', 0.46704691648483276), ('turf', 0.46331942081451416), ('trees', 0.4560699760913849), ('ground', 0.45392271876335144)]
print(glove.most_similar("laptop"))
#[('laptops', 0.7956499457359314), ('computers', 0.6733037233352661), ('phones', 0.599344789981842), ('computer', 0.5955509543418884), ('portable', 0.5796298384666443), ('desktop', 0.5617854595184326), ('cellphones', 0.5468271970748901), ('notebooks', 0.5464344024658203), ('pcs', 0.5435254573822021), ('cellphone', 0.5287015438079834)]
print(glove.most_similar("lamp"))
#[('lamps', 0.8024932146072388), ('bulb', 0.5680024027824402), ('candle', 0.5575706958770752), ('incandescent', 0.5573796629905701), ('fluorescent', 0.5541897416114807), ('lights', 0.5355976819992065), ('kerosene', 0.513739287853241), ('halogen', 0.5012800693511963), ('lighting', 0.4974631071090698), ('lighted', 0.48428040742874146)]
print(glove.most_similar("globe"))
#[('columnist', 0.4608248472213745), ('times', 0.42327189445495605), ('boston', 0.40685129165649414), ('cox', 0.39769992232322693), ('mail', 0.39709267020225525), ('reporter', 0.3864489495754242), ('awards', 0.3778129816055298), ('wbz', 0.37551382184028625), ('correspondent', 0.37502551078796387), ('newspapers', 0.3736392557621002)]
print(glove.most_similar("bird"))
#[('birds', 0.7303513288497925), ('flu', 0.7103857398033142), ('avian', 0.6787645220756531), ('h5n1', 0.6514254212379456), ('influenza', 0.58339524269104), ('virus', 0.5779675841331482), ('migratory', 0.568098247051239), ('swine', 0.5425065159797668), ('poultry', 0.5381720662117004), ('outbreaks', 0.5357742309570312)]
print(glove.most_similar("bee"))
#[('bees', 0.5616937279701233), ('gees', 0.5614164471626282), ('gee', 0.46755173802375793), ('honey', 0.45776888728141785), ('hive', 0.4324120879173279), ('jehn', 0.411525160074234), ('eby', 0.40299704670906067), ('tul', 0.4023614823818207), ('bumble', 0.40058445930480957), ('hives', 0.3977218568325043)]
