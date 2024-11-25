import pandas as pd
import numpy as np



# Normalizzazione dei nomi delle province
def normalizeNameProvince(a):
    return a.strip().lower().replace("'", "").replace(" ", "")

def aggiungiProvinceMancanti(dictProvinceRegioni):
    province_mancanti = {
        'agrigento': 'Sicilia',
        'alessandria': 'Piemonte',
        'ancona': 'Marche',
        'arezzo': 'Toscana',
        'ascolipiceno': 'Marche',
        'asti': 'Piemonte',
        'avellino': 'Campania',
        'bari': 'Puglia',
        'barletta-andria-trani': 'Puglia',
        'belluno': 'Veneto',
        'benevento': 'Campania',
        'bergamo': 'Lombardia',
        'biella': 'Piemonte',
        'bologna': 'Emilia-Romagna',
        'bolzano': 'Trentino-Alto Adige',
        'brescia': 'Lombardia',
        'brindisi': 'Puglia',
        'cagliari': 'Sardegna',
        'caltanissetta': 'Sicilia',
        'campobasso': 'Molise',
        'caserta': 'Campania',
        'catania': 'Sicilia',
        'catanzaro': 'Calabria',
        'chieti': 'Abruzzo',
        'como': 'Lombardia',
        'cosenza': 'Calabria',
        'cremona': 'Lombardia',
        'crotone': 'Calabria',
        'cuneo': 'Piemonte',
        'enna': 'Sicilia',
        'fermo': 'Marche',
        'ferrara': 'Emilia-Romagna',
        'firenze': 'Toscana',
        'foggia': 'Puglia',
        'forlì-cesena': 'Emilia-Romagna',
        'frosinone': 'Lazio',
        'genova': 'Liguria',
        'gorizia': 'Friuli-Venezia Giulia',
        'grosseto': 'Toscana',
        'imperia': 'Liguria',
        'isernia': 'Molise',
        'laquila': 'Abruzzo',
        'laspezia': 'Liguria',
        'latina': 'Lazio',
        'lecce': 'Puglia',
        'lecco': 'Lombardia',
        'livorno': 'Toscana',
        'lodi': 'Lombardia',
        'lucca': 'Toscana',
        'macerata': 'Marche',
        'mantova': 'Lombardia',
        'massa-carrara': 'Toscana',
        'matera': 'Basilicata',
        'messina': 'Sicilia',
        'milano': 'Lombardia',
        'modena': 'Emilia-Romagna',
        'monzaebrianza': 'Lombardia',
        'napoli': 'Campania',
        'novara': 'Piemonte',
        'nuoro': 'Sardegna',
        'oristano': 'Sardegna',
        'padova': 'Veneto',
        'palermo': 'Sicilia',
        'parma': 'Emilia-Romagna',
        'pavia': 'Lombardia',
        'perugia': 'Umbria',
        'pesaroeurbino': 'Marche',
        'pescara': 'Abruzzo',
        'piacenza': 'Emilia-Romagna',
        'pisa': 'Toscana',
        'pistoia': 'Toscana',
        'pordenone': 'Friuli-Venezia Giulia',
        'potenza': 'Basilicata',
        'prato': 'Toscana',
        'ragusa': 'Sicilia',
        'ravenna': 'Emilia-Romagna',
        'reggiocalabria': 'Calabria',
        'reggioemilia': 'Emilia-Romagna',
        'rieti': 'Lazio',
        'rimini': 'Emilia-Romagna',
        'roma': 'Lazio',
        'rovigo': 'Veneto',
        'salerno': 'Campania',
        'sassari': 'Sardegna',
        'savona': 'Liguria',
        'siena': 'Toscana',
        'siracusa': 'Sicilia',
        'sondrio': 'Lombardia',
        'sudsardegna': 'Sardegna',
        'taranto': 'Puglia',
        'teramo': 'Abruzzo',
        'terni': 'Umbria',
        'torino': 'Piemonte',
        'trapani': 'Sicilia',
        'trento': 'Trentino-Alto Adige',
        'treviso': 'Veneto',
        'trieste': 'Friuli-Venezia Giulia',
        'udine': 'Friuli-Venezia Giulia',
        'valledaosta': 'Valle d\'Aosta',
        'varese': 'Lombardia',
        'venezia': 'Veneto',
        'verbano-cusio-ossola': 'Piemonte',
        'vercelli': 'Piemonte',
        'verona': 'Veneto',
        'vibovalentia': 'Calabria',
        'vicenza': 'Veneto',
        'viterbo': 'Lazio'
    }
    
    
    dictProvinceRegioni.update(province_mancanti)
    
    return dictProvinceRegioni


def readFluDatasetInflucast(flu_file):
    # Popolazione e classificazione
    pop_df = pd.read_csv('Data\\FinalForCommuting\\pop.csv')
    province_classification_df = pd.read_csv('Data\\General\\ProvincieRegioneClassificazione.csv')
    flowMatrix = pd.read_csv('Data\\FinalForCommuting\\A_adj_province.csv', sep=";", index_col=0)
    
    # Normalizzazione dei nomi delle province
    pop_df['Territorio'] = pop_df['Territorio'].apply(normalizeNameProvince)
    popPROV = dict(zip(pop_df['Territorio'], pop_df['Value']))
    orderedPROV = pop_df['Territorio'].tolist()
    
    # Matrice dei flussi
    flowMatrix.index = flowMatrix.index.to_series().apply(normalizeNameProvince)
    flowMatrix.columns = flowMatrix.columns.to_series().apply(normalizeNameProvince)
    flowMatrix = flowMatrix.loc[orderedPROV, orderedPROV]
    Whk = flowMatrix.to_numpy()
    np.fill_diagonal(Whk, 0)  
    Whk = Whk / Whk.sum(1).reshape(-1, 1)  # Normalizzazione
    
    # Caricamento dei dati di influenza iniziale
    flu_df = pd.read_csv(flu_file)
    flu_df['Territorio'] = flu_df['Territorio'].apply(normalizeNameProvince)
    
    # Parsing della colonna 'Settimana'
    flu_df['Anno'] = flu_df['Settimana'].str.split('_W').str[0].astype(int)
    flu_df['Settimana'] = flu_df['Settimana'].str.split('_W').str[1].astype(int)
    
    # Calcolo della settimana più recente per ogni provincia
    flu_df = flu_df.sort_values(by=['Anno', 'Settimana'], ascending=True)  # Ordina per anno e settimana
    recent_flu_df = flu_df.groupby('Territorio').last().reset_index()  # Prendi la settimana più recente per provincia
    
    # Mappare i casi iniziali
    flu_init_dict = dict(zip(recent_flu_df['Territorio'], recent_flu_df['CasiDistribuiti']))
    nh = np.array([popPROV[prov] for prov in orderedPROV])
    initial_case_i = np.array([flu_init_dict.get(prov, 0) for prov in orderedPROV])
    initial_case_e = np.zeros_like(initial_case_i)  # Supponiamo che non ci siano esposti inizialmente
    
    return nh, Whk, initial_case_i, initial_case_e, orderedPROV, flu_init_dict, flu_df, recent_flu_df


def readFluDataset1():
    
    pop_df = pd.read_csv('Data\\FinalForCommuting\\pop.csv')
    flu_init_df = pd.read_csv('flu_prov_init17-18.csv') # seleziona il file con l'anno da cui desideri partire
    province_classification_df = pd.read_csv('Data\\General\\ProvincieRegioneClassificazione.csv')
    flowMatrix = pd.read_csv('Data\\FinalForCommuting\\A_adj_province.csv', sep=";", index_col=0)

    
    pop_df['Territorio'] = pop_df['Territorio'].apply(normalizeNameProvince)
    popPROV = dict(zip(pop_df['Territorio'], pop_df['Value']))

    
    orderedPROV = pop_df['Territorio'].tolist()
    flowMatrix.index = flowMatrix.index.to_series().apply(normalizeNameProvince)
    flowMatrix.columns = flowMatrix.columns.to_series().apply(normalizeNameProvince)
    flowMatrix = flowMatrix.loc[orderedPROV, orderedPROV]

    
    Whk = flowMatrix.to_numpy()
    np.fill_diagonal(Whk, 0)  
    Whk = Whk / Whk.sum(1).reshape(-1, 1)  # Normalizzazione

    nh = np.array([popPROV[prov] for prov in orderedPROV])

    flu_init_df['provincia'] = flu_init_df['provincia'].apply(normalizeNameProvince)
    flu_init_dict = dict(zip(flu_init_df['provincia'], flu_init_df['numero_infetti']))
    initial_case_i = np.array([flu_init_dict.get(prov, 0) for prov in orderedPROV])
    initial_case_e = np.zeros_like(initial_case_i)  # Supponiamo che non ci siano esposti inizialmente
    return nh, Whk, initial_case_i, initial_case_e, orderedPROV, flu_init_dict


class parameters(): #parametri che dovremo variare
    def __init__(self, nh, Whk):
        self.nh = nh
        self.Whk = Whk
        self.ai = np.array([0.149, 0.545, 0.545])  # Livelli di attività: [giovani, anziani, anziani vaccinati. Va preso il reciproco perché l'unità di misura è 1/days]
        #self.ai = np.array([1.0, 0.5, 0.5])
        self.eta = np.array([0.5, 0.25, 0.25])  # Proporzione di popolazione in ciascuna classe iniziale
        #self.b = 0.002  # Tasso di contatto tra province in un caso estremo di lockdown draconiano
        #self.b = 0.02 # Tasso di contatto tra province con viaggi fortemente ridotti
        self.b = 0.09 # parametrò di mobilità
        
        self.Lambda = np.array([1, 1.3, 1.2]) * (10**-8) * 5.33 # Tasso di trasmissione
        #self.Nu = 0.1  # Tasso di infezione da esposto a infetto
        #self.Beta = 0.05  # I -> N
        #self.Gamma = 0.01  # Tasso di rimozione (guariti)

        self.Nu = 0.266 
        self.Beta = 0.4 * 6 
        self.Gamma = 0.1
        self.alpha = np.array([1.0, 0.8, 0.9])  # Efficacia dell'autoisolamento: [giovani, anziani, anziani vaccinati]
        self.m = 19.77  # numero medio di contatti