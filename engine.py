import pandas as pd
import numpy as np
from bidict import bidict
import copy

def normalizeNameProvince(a):
    if a == "Reggio di Calabria": a = "Reggio Calabria"
    if "Aosta" in a: a = "Valle d'Aosta"
    if "Bolzano" in a: a = "Bolzano"
    if "Trento" in a: a = "Trento"
    if "Trentino" in a: a = "Trentino"
    if "Friuli-Venezia Giulia" in a: a = "Friuli Venezia Giulia"
    if a == "Massa Carrara": a = "Massa-Carrara"
    return a

def readFluDataset(fluProvPath = None, covidRegionPath = None, summerW=False):
    
    if fluProvPath is None: fluProvPath = ".\\flu_prov_init.csv"
    if covidRegionPath is None: covidRegionPath = ".\\Data\\Epidemia\\dpc-covid19-ita-regioni.csv"
        
    #############################################
    # MAP province to north-south-centre
    #############################################
    provincie = pd.read_csv(".\\Data\\General\\ProvincieRegioneClassificazione.csv", sep=";", header=[0,1])
    provincie = provincie.droplevel(0, axis=1) # mantiene solo le prime due colonne, con nome provincia e regione
    provincie = provincie[provincie.TIPO_UTS.isin(['Citta metropolitana', 'Libero consorzio di comuni', 'Provincia', 'Provincia autonoma', "Unità non amministrativa"])]
    provincie.DEN_UTS = provincie.DEN_UTS.apply(normalizeNameProvince) #normalizza nomi province
    dictProvincieNS = dict(provincie[["DEN_UTS", "DEN_RIP"]].to_numpy()) #crea un dizionario che associa a ogni provincia nord, centro o sud

    
    #############################################
    # MAP region to north-south-centre
    #############################################
    provincie = pd.read_csv(".\\Data\\General\\ProvincieRegioneClassificazione.csv", sep=";", header=[0,1])
    provincie = provincie.droplevel(0, axis=1).drop_duplicates("DEN_REG")
    provincie.DEN_REG = provincie.DEN_REG.apply(normalizeNameProvince)
    dictRegionNS = dict(provincie[["DEN_REG", "DEN_RIP"]].to_numpy())

    
    #############################################
    # Population and ordered province
    #############################################
    pop = pd.read_csv(".\\Data\\FinalForCommuting\\pop.csv", sep=",", index_col=None)
    orderedPROV = pop.Territorio.to_numpy()
    popPROV = pop.set_index("Territorio").Value.to_dict() # dizionario che associa ogni provincia alla sua popolazione
    indexPROV = bidict(zip(orderedPROV, range(len(orderedPROV)))) # dizionario che associa ogni provincia al suo numero
    nh = pop.Value.to_numpy() # vettore contenente popolazione di ogni provincia

    
    #############################################
    # Travel matrix
    #############################################
    flowMatrix = pd.read_csv(".\\Data\\FinalForCommuting\\A_adj_province.csv", sep=";", index_col=0) # specifica che la prima colonna contiene gli indici delle province
        
    flowMatrix = flowMatrix.loc[orderedPROV, orderedPROV] # riordina righe e colonne seguendo l'ordine di pop.csv
    Whk = flowMatrix.to_numpy()
    np.fill_diagonal(Whk, 0) # azzera la diagonale principale, non ci interessano i movimenti intra provinciali
    Whk = (Whk / Whk.sum(1).reshape(-1, 1)) # normalizza a 1

    
    #############################################
    # Dict Provincia --> Regione
    #############################################
    province = pd.read_csv(".\\Data\\General\\ProvincieRegioneClassificazione.csv", sep=";", header=[0,1])
    province = province.droplevel(0, axis=1)
    province = province[province.TIPO_UTS.isin(['Citta metropolitana', 'Libero consorzio di comuni', 'Provincia', 'Provincia autonoma', "Unità non amministrativa"])]
    province.DEN_UTS = province.DEN_UTS.apply(normalizeNameProvince)
    province.DEN_REG = province.DEN_REG.apply(normalizeNameProvince)
    convertPROV_REG = dict(province[["DEN_UTS", "DEN_REG"]].to_numpy())

    # sintetizzando, province contiene i nomi normalizzati delle province, convertPROV_REG è un dizionario che associa ogni provincia alla sua regione normalizzata
    

    # prende il dataset provinciale al tempo zero, e lo salva nella variabile fluProv per poterlo usare come condizione iniziale
    fluProv = pd.read_csv(fluProvPath)
    fluProv.denominazione_provincia = fluProv.provincia.apply(normalizeNameProvince)
    # Creazione di una colonna datetime a partire dalle colonne Anno e Settimana
    fluProv['data'] = fluProv.apply(lambda row: pd.to_datetime(f"{row['anno']}-W{int(row['settimana'])}-1", format="%G-W%V-%u").strftime('%d-%m-%Y'), axis=1)
    # Impostiamo la colonna 'provincia' come categorica con un ordine specifico
    fluProv['provincia'] = pd.Categorical(fluProv['provincia'], categories=orderedPROV, ordered=True)

    # Riordinamento del dataframe in base alla colonna 'provincia'
    fluProv = fluProv.sort_values('provincia')

    #############################################
    # Covid Dataset Province
    #############################################
    
    
    

    #covidProv.data = pd.to_datetime(covidProv.data)
    #covidProv = covidProv[["data", "denominazione_provincia", "totale_casi"]]
    #covidProv = covidProv[~(covidProv.denominazione_provincia == "In fase di definizione/aggiornamento")]
    #covidProv = covidProv[~(covidProv.denominazione_provincia == "Fuori Regione / Provincia Autonoma")]

    #covidProv = covidProv.set_index(["data", "denominazione_provincia"]).unstack().resample('1D').max()
    #covidProv = covidProv.droplevel(0, axis=1)

    #covidProv = covidProv[orderedPROV]

    
    #############################################
    # Covid Dataset Regioni
    #############################################
    
    # ci manca da salvare tutto il dataset covid regionale per poterlo usare come benchmark del modello. Conviene trasformarlo in provinciale per le stime bayesiane? 

    #covidRegion = pd.read_csv(covidRegionPath)
    #covidRegion.data = pd.to_datetime(covidRegion.data)
    #covidRegion = covidRegion[["data", "denominazione_regione", "nuovi_positivi"]]

    #covidRegion = covidRegion.set_index(["data", "denominazione_regione"]).unstack().resample('1D').max()
    #covidRegion = covidRegion.droplevel(0, axis=1)


    # -- cumulate
    #covidRegionCum = covidRegion.cumsum(axis=0)
    #covidRegionCum["Trentino"] = covidRegionCum[["P.A. Bolzano","P.A. Trento"]].sum(1)
    #covidRegionCum = covidRegionCum.drop(["P.A. Bolzano","P.A. Trento"], axis=1)


    # -- new I
    #covidRegionNew = covidRegion
    #covidRegionNew["Trentino"] = covidRegionNew[["P.A. Bolzano","P.A. Trento"]].sum(1)
    #covidRegionNew = covidRegionNew.drop(["P.A. Bolzano","P.A. Trento"], axis=1)


    # -- actual I
    #covidRegion = pd.read_csv(".\\Data\\Epidemia\\dpc-covid19-ita-regioni.csv")
    #covidRegion.data = pd.to_datetime(covidRegion.data)
    #covidRegion = covidRegion[["data", "denominazione_regione", "totale_positivi"]]
    #orderedREG = list(set(covidRegion.denominazione_regione))

    #covidRegion = covidRegion.set_index(["data", "denominazione_regione"]).unstack().resample('1D').max()
    #covidRegion = covidRegion.droplevel(0, axis=1)

    #covidRegion["Trentino"] = covidRegion[["P.A. Bolzano","P.A. Trento"]].sum(1)
    #covidRegion = covidRegion.drop(["P.A. Bolzano","P.A. Trento"], axis=1)

    # Account for negative values
    #covidRegionNew = covidRegionNew.map(lambda a: max(a,0))
    

    return fluProv

class parameters(): #parametri che dovremo variare
    def __init__(self, nh, Whk):
        self.nh = nh
        self.Whk = Whk
        self.ai = np.array([1.0, 0.5, 0.3])  # Livelli di attività: [giovani, anziani, anziani vaccinati]
        self.eta = np.array([0.5, 0.3, 0.2])  # Proporzione di popolazione in ciascuna classe iniziale
        self.b = 0.2  # Tasso di contatto tra province
        self.Lambda = 0.3  # Tasso di trasmissione
        self.Nu = 0.1  # Tasso di infezione da esposto a infetto
        self.Beta = 0.05  # Tasso di recupero/morte
        self.Gamma = 0.01  # Tasso di rimozione (guariti)
        self.alpha = np.array([1.0, 0.8, 0.9])  # Efficacia dell'autoisolamento: [giovani, anziani, anziani vaccinati]
        self.m = 0.9  # Tasso di mobilità