# import dask.dataframe as dd
import matplotlib
import csv
import matplotlib.pyplot as plt
# import miRBaseFunctions
import numpy as np
import os
import pandas as pd
import pickle
import scipy.cluster.hierarchy as SciPyClus
from singscore.singscore import *
from sklearn.neighbors import KNeighborsClassifier
# import TCGAFunctions
import urllib

class PathDir:
    # A module to handle paths (hopefully OS independent); if users wish to manually download the DepMap data then
    #  these paths can be edited/hard-coded to the appropriate location
    #   #   #   #   #   #

    pathScriptDir = os.getcwd()
    # pathBaseDir = 'C:\\Users\\jcur0014\\OneDrive - Monash University\\papers\\2021_Sgro_LIHC_methyl_effects'

    pathDataFolder = os.path.join(pathScriptDir, 'data')
    if not os.path.exists(pathDataFolder):
        os.mkdir(pathDataFolder)

    pathPlotFolder = os.path.join(pathScriptDir, 'figures')
    if not os.path.exists(pathPlotFolder):
        os.mkdir(pathPlotFolder)

    # strHepaRGDNAmeLoc = 'D:\\db\\geo\\gse72074_-_HepaRG_DNAme'
    # strHepaRGDNAmeFile = 'GSE72074_betas_raw.txt'
    #
    # strOrganoidDataLoc = 'D:\\db\\geo\\gse84073_-_liver_organoids'
    #
    # strAdultVsFetalDNAmeLoc = 'D:\\db\\geo\\gse61278_-_fetal_vs_adult'
    # strAdultVsFetalDNAmeFile = 'GSE61278_MatrixProcessed.txt'


class PreProc:

    def download_missing_file(url, filepath):
        # a function to download files from specified paths as required

        if not os.path.exists(filepath):
            print(f'Cannot detect the following file:\n\t\t{filepath}'
                  f'\n\tDownloading via figshare, this may take some time..')
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                with open(filepath, 'wb') as outfile:
                    outfile.write(response.read())
        return ()

    def inhouse_dname(flagResult=False):


        dfHHIPProbes = pd.read_csv(os.path.join(PathDir.pathDataFolder, 'HHIP_beta.tsv'),
                                   sep='\t', index_col=0, header=0)
        listHHIPProbes = dfHHIPProbes.index.tolist()

        listFiles = ['AllProbes_limma_SWAN_A_NoG.txt', 'AllProbes_limma_SWAN_i_NoG.txt']
        listCompNames = ['ConstructOne','ConstructTwo']
        listToMerge = []
        for iFile in range(len(listFiles)):
            dfIn = pd.read_csv(os.path.join(PathDir.pathPlotFolder, listFiles[iFile]),
                                         sep='\t', index_col=0, header=0)
            dfToCopy = dfIn[['logFC', 'adj.P.Val']].reindex(listHHIPProbes).copy(deep=True)
            listHeaders = dfToCopy.columns.tolist()
            listHeadersNew = [listCompNames[iFile]+':'+strHeader for strHeader in listHeaders]
            dfToCopy.rename(columns=dict(zip(listHeaders, listHeadersNew)),inplace=True)

            listToMerge.append(dfToCopy)

        dfOut = pd.concat(listToMerge, axis=1)

        return dfOut

class TCGAFunctions:

    # https://gdc.cancer.gov/about-data/publications/pancanatlas

    def lihc_metadata(flagResult=False,
                           flagPerformExtraction=False):

        pathTempFile = os.path.join(PathDir.pathDataFolder, 'TCGA-LIHC_metadata.pickle')

        if not os.path.exists(pathTempFile):
            flagPerformExtraction = True

        if flagPerformExtraction:

            dfTCGAAnnot = pd.read_excel(os.path.join(PathDir.pathDataFolder,
                                                 'TCGA-CDR-SupplementalTableS1.xlsx'),
                                    header=0, index_col=0, sheet_name='TCGA-CDR')

            dfLIHCMetaData = dfTCGAAnnot[dfTCGAAnnot['type']=='LIHC']
            dfLIHCMetaData.set_index('bcr_patient_barcode', inplace=True)
            listLIHCOutIndex = dfLIHCMetaData.index.tolist()

            dfLIHCPaperAnnot = pd.read_excel(
                os.path.join(PathDir.pathDataFolder, 'mmc1.xlsx'),
                sheet_name='Table S1A - core sample set',
                skiprows=3,
                index_col='Barcode')
            sliceClusterData = dfLIHCPaperAnnot['iCluster clusters (k=3, Ronglai Shen)']

            arrayCleanClusterData = np.zeros(len(listLIHCOutIndex), dtype=np.int)
            for iSample in range(len(listLIHCOutIndex)):
                strSample = listLIHCOutIndex[iSample] + '-01A'

                if strSample in sliceClusterData.index.tolist():
                    if sliceClusterData.loc[strSample] == sliceClusterData.loc[strSample]:
                        strCluster = sliceClusterData.loc[strSample]
                        arrayCleanClusterData[iSample] = np.int(strCluster.split('iCluster:')[1])
                    else:
                        arrayCleanClusterData[iSample] = 0
                else:
                    arrayCleanClusterData[iSample] = 0

            dfLIHCMetaData['iCluster'] = pd.Series(arrayCleanClusterData,
                                                   index=listLIHCOutIndex)

            dfLIHCMetaData.to_pickle(pathTempFile)

        else:
            dfLIHCMetaData = pd.read_pickle(pathTempFile)

        return dfLIHCMetaData

    def lihc_mess_rna(flagResult=False,
                      flagPerformExtraction=False):

        pathTempFile = os.path.join(PathDir.pathDataFolder, 'TCGA-LIHC_RNA.pickle')

        if not os.path.exists(pathTempFile):
            flagPerformExtraction=True

        if flagPerformExtraction:
            dfMetaLIHC = TCGAFunctions.lihc_metadata()
            setLIHCPatients = set(dfMetaLIHC.index.tolist())

            # http://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611

            print('Attempting to load processed TCGA pan-cancer RNA-seq data')
            print('.. this is an approx. 2GB text file and may take several minutes..')
            strFileName = 'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv'
            dfRNAHeader = pd.read_csv(
                os.path.join(PathDir.pathDataFolder, strFileName),
                sep='\t', header=0, index_col=0, nrows=0)
            listColumns = dfRNAHeader.columns.tolist()
            listLIHCSamples = [strSample for strSample in listColumns
                               if strSample[0:len('TCGA-NN-NNNN')] in setLIHCPatients]

            dfRNA = pd.read_csv(
                os.path.join(PathDir.pathDataFolder, strFileName),
                sep='\t', header=0, index_col=0,
                usecols=['gene_id']+listLIHCSamples,
                dtype=dict(zip(listLIHCSamples, [float]*len(listLIHCSamples))))

            # there were previously two genes known as SLC35E2; these are now renamed as
            #  SLC35E2A/B to prevent issues with mapping to HGNC symbols
            dfRNA.rename(index={'SLC35E2|9906':'SLC35E2A|9906',
                                    'SLC35E2|728661':'SLC35E2B|728661'},
                             inplace=True)

            print('\t.. applying log transformation to abundance data')
            arrayLogData = np.log2(dfRNA.values.astype(float)+1)
            dfRNA.iloc[:,:] = arrayLogData

            print('\t.. saving intermediate files for later reuse')
            dfRNA.to_pickle(pathTempFile)

        else:
            print('Loading pre-processed log-transformed RNA-seq data')
            dfRNA = pd.read_pickle(pathTempFile)

        return dfRNA

    def lihc_micro_rna(flagResult=False,
                       flagPerformExtraction=False):

        pathTempFile = os.path.join(PathDir.pathDataFolder, 'TCGA-LIHC_miRNA.pickle')

        if not os.path.exists(pathTempFile):
            flagPerformExtraction=True

        if flagPerformExtraction:
            dfMetaLIHC = TCGAFunctions.lihc_metadata()
            setLIHCPatients = set(dfMetaLIHC.index.tolist())

            # http://api.gdc.cancer.gov/data/3586c0da-64d0-4b74-a449-5ff4d9136611

            print('Attempting to load processed TCGA pan-cancer miRNA-seq data')
            strFileName = 'pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv'
            dfRNAHeader = pd.read_csv(
                os.path.join(PathDir.pathDataFolder, strFileName),
                sep=',', header=0, index_col=0, nrows=0)
            listColumns = dfRNAHeader.columns.tolist()
            listColumns.remove('Correction')
            listLIHCSamples = [strSample for strSample in listColumns
                               if strSample[0:len('TCGA-NN-NNNN')] in setLIHCPatients]

            dfRNA = pd.read_csv(
                os.path.join(PathDir.pathDataFolder, strFileName),
                sep=',', header=0, index_col=0,
                usecols=['Genes']+listLIHCSamples,
                dtype=dict(zip(listLIHCSamples, [float]*len(listLIHCSamples))))


            print('\t.. applying log transformation to abundance data')
            arrayLogData = np.log2(dfRNA.values.astype(float)+1)
            dfRNA.iloc[:,:] = arrayLogData

            print('\t.. saving intermediate files for later reuse')
            dfRNA.to_pickle(pathTempFile)

        else:
            print('Loading pre-processed log-transformed RNA-seq data')
            dfRNA = pd.read_pickle(pathTempFile)

        return dfRNA

    def lihc_dna_me(flagResult=False,
                    flagPerformExtraction=False):

        pathTempFile = os.path.join(PathDir.pathDataFolder, 'methylation_preproc.pickle')
        pathTempMappingFile = os.path.join(PathDir.pathDataFolder, 'methylation_mapping.pickle')

        if not os.path.exists(pathTempFile):
            flagPerformExtraction = True

        if flagPerformExtraction:
            dfIn = pd.read_csv(os.path.join(PathDir.pathDataFolder, 'methylation_data.txt'),
                               sep='\t')
            listSampleBarcodes = dfIn.columns.tolist()
            listSampleBarcodes.remove('Hybridization REF')
            listSampleBarcodesClean = [strBarcode.split('.')[0] for strBarcode in listSampleBarcodes]

            listColumnType = dfIn.iloc[0, :].tolist()
            arrayBetaValueColumnIndices = np.where([strColumn == 'Beta_value' for strColumn in listColumnType])[0]
            arrayGeneSymbolColumnIndices = np.where([strColumn == 'Gene_Symbol' for strColumn in listColumnType])[0]
            arrayGenCoordColumnIndices = \
            np.where([strColumn == 'Genomic_Coordinate' for strColumn in listColumnType])[0]
            arrayChromColumnIndices = np.where([strColumn == 'Chromosome' for strColumn in listColumnType])[0]

            listCompEleRef = dfIn.iloc[1:, 0].tolist()

            listUniqueBarcodes = []
            for strBarcode in listSampleBarcodesClean:
                if strBarcode not in listUniqueBarcodes:
                    listUniqueBarcodes.append(strBarcode)

            dfData = pd.DataFrame(columns=listUniqueBarcodes,
                                  index=listCompEleRef,
                                  data=dfIn.iloc[1:, arrayBetaValueColumnIndices].values.astype(float))

            dictOfDictsProbeMapping = {}
            dictOfDictsProbeMapping['Gene'] = dict(zip(listCompEleRef,
                                                       dfIn.iloc[1:, arrayGeneSymbolColumnIndices[0]].values))
            dictOfDictsProbeMapping['GenCoord'] = dict(zip(listCompEleRef,
                                                           dfIn.iloc[1:, arrayGenCoordColumnIndices[0]].values))
            dictOfDictsProbeMapping['Chr'] = dict(zip(listCompEleRef,
                                                      dfIn.iloc[1:, arrayChromColumnIndices[0]].values))

            dfData.to_pickle(pathTempFile)

            with open(pathTempMappingFile, 'wb') as handFile:
                pickle.dump(dictOfDictsProbeMapping, handFile, protocol=pickle.HIGHEST_PROTOCOL)


        else:
            dfData = pd.read_pickle(pathTempFile)

            with open(pathTempMappingFile, 'rb') as handFile:
                dictOfDictsProbeMapping = pickle.load(handFile)

        return dfData, dictOfDictsProbeMapping

    def lihc_copy_number(flagResult=False,
                         flagPerformExtraction=False):

        pathTempFile = os.path.join(PathDir.pathDataFolder, 'TCGA-LIHC_CNV.pickle')

        if not os.path.exists(pathTempFile):
            flagPerformExtraction=True

        if flagPerformExtraction:
            dfMetaLIHC = TCGAFunctions.lihc_metadata()
            setLIHCPatients = set(dfMetaLIHC.index.tolist())

            # http://api.gdc.cancer.gov/data/00a32f7a-c85f-4f86-850d-be53973cbc4d

            print('Attempting to load processed TCGA pan-cancer CNV data')
            strFileName = 'broad.mit.edu_PANCAN_Genome_Wide_SNP_6_whitelisted.seg'

            listSamples = []
            listChr = []
            listStart = []
            listEnd = []
            listSegMean = []

            flagHeader=True
            with open(os.path.join(PathDir.pathDataFolder, strFileName)) as handFile:
                for listRow in csv.reader(handFile, delimiter='\t'):
                    if flagHeader:
                        flagHeader = False
                    if listRow[0][0:len('TCGA-NN-NNNN')] in setLIHCPatients:
                        listSamples.append(listRow[0])
                        listChr.append(listRow[1])
                        listStart.append(float(listRow[2]))
                        listEnd.append(float(listRow[3]))
                        listSegMean.append(float(listRow[5]))

            dfCNV = pd.DataFrame({'Sample':listSamples,
                                  'Chromosome':listChr,
                                  'Start':listStart,
                                  'End':listEnd,
                                  'Segment_Mean':listSegMean},
                                 index=np.arange(len(listSamples)))

            print('\t.. saving intermediate files for later reuse')
            dfCNV.to_pickle(pathTempFile)

        else:
            print('Loading pre-processed CNV data')
            dfCNV = pd.read_pickle(pathTempFile)

        return dfCNV

    def lihc_mutation(flagResult=False,
                      flagPerformExtraction=False):

        pathTempFile = os.path.join(PathDir.pathDataFolder, 'TCGA-LIHC_Mut.pickle')

        if not os.path.exists(pathTempFile):
            flagPerformExtraction = True

        if flagPerformExtraction:
            dfMetaLIHC = TCGAFunctions.lihc_metadata()
            setLIHCPatients = set(dfMetaLIHC.index.tolist())

            # http://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc

            print('Attempting to load processed TCGA pan-cancer mutation data'
                  '\n.. this is a ~730MB gzipped text file and may take some time')
            strFileName = 'mc3.v0.2.8.PUBLIC.maf.gz'
            # relatively large table, dask seems to dislike the gzipped file, just use pandas
            dfMutSamples = pd.read_csv(os.path.join(PathDir.pathDataFolder, strFileName),
                                 compression='gzip', sep='\t', usecols=['Tumor_Sample_Barcode'])

            listSamples = dfMutSamples['Tumor_Sample_Barcode'].values.tolist()
            arraySampleIsLIHC = np.array(
                [strSample[0:len('TCGA-NN-NNNN')] in setLIHCPatients for strSample in listSamples],
                dtype=np.bool)
            arrayRowsToSkip = np.where(~arraySampleIsLIHC)[0]

            dfMut = pd.read_csv(
                os.path.join(PathDir.pathDataFolder, strFileName),
                sep='\t', header=0, index_col=None,
                compression='gzip',
                usecols=['Hugo_Symbol', 'Entrez_Gene_Id', 'dbSNP_RS',
                         'Tumor_Sample_Barcode', 'HGVSp_Short', 'Gene',
                         'IMPACT', 'PolyPhen'],
                skiprows=arrayRowsToSkip+1)

            print('\t.. saving intermediate files for later reuse')
            dfMut.to_pickle(pathTempFile)

        else:
            print('Loading pre-processed mutation data')
            dfMut = pd.read_pickle(pathTempFile)

        return dfMut

    def infer_unclassified_clusters(flagResult=False,
                                    flagPerformInference=False,
                                    flagPlotInfClust=True):

        pathTempFile = os.path.join(PathDir.pathDataFolder, 'TCGA-LIHC-infer_clust.pickle')

        if not os.path.exists(pathTempFile):
            flagPerformInference=True

        if flagPerformInference:

            dfMessRNA = TCGAFunctions.lihc_mess_rna()
            listSamples = dfMessRNA.columns.tolist()
            listMessRNAs = dfMessRNA.index.tolist()
            listMessRNAsClean = []
            for strMessRNA in listMessRNAs:
                if strMessRNA.startswith('?|'):
                    listMessRNAsClean.append(strMessRNA)
                else:
                    listMessRNAsClean.append(strMessRNA.split('|')[0])
            dfMessRNA.index = listMessRNAsClean

            listTumourSamples = [strSample for strSample in listSamples if strSample[13:15]=='01']

            dfMetaLIHC = TCGAFunctions.lihc_metadata()

            dictGeneSets = TCGAFunctions.int_clust_genesets()

            arrayGeneSetScores = np.zeros((len(listTumourSamples),3), dtype=float)

            print('Scoring TCGA HCC samples for iCluster gene sets..')
            for iSample in range(len(listTumourSamples)):
            # for iSample in range(50):
                print('\t .. ' + '{}'.format(iSample+1) + ' of ' + '{}'.format(len(listTumourSamples)))
                strSample = listTumourSamples[iSample]

                for iClust in range(3):
                    dfScore = score(up_gene=dictGeneSets[f'IntClust{iClust+1}']['UpGenes'],
                                    down_gene=dictGeneSets[f'IntClust{iClust+1}']['DownGenes'],
                                    sample=dfMessRNA[[strSample]])
                    arrayGeneSetScores[iSample, iClust] = dfScore['total_score'].values.astype(float)[0]

            listTumourSamplesClean = [strSample[0:len('TCGA-NN-NNNN-NNA')] for strSample in listTumourSamples]

            dfGeneSetScores = pd.DataFrame(data=arrayGeneSetScores,
                                           index=listTumourSamplesClean,
                                           columns=['iClust1 score', 'iClust2 score', 'iClust3 score'])

            listKnownClassSamples = []
            listKnownClasses = []
            listOfListsClusterPatients = []
            for iClust in range(3):
                listPatientsInCluster = dfMetaLIHC[dfMetaLIHC['iCluster']==iClust+1].index.tolist()
                listSamplesOfInt = [strPatient + '-01A' for strPatient in listPatientsInCluster]
                listKnownClassSamples += listSamplesOfInt
                listOfListsClusterPatients.append(listSamplesOfInt)
                listKnownClasses += [iClust+1]*len(listPatientsInCluster)

            arrayKnownClasses = np.array(listKnownClasses, dtype=float)
            arrayKnownClassScores = dfGeneSetScores.reindex(listKnownClassSamples).values.astype(float)

            listUnclassified = [strSample for strSample in listTumourSamplesClean
                                if strSample not in listKnownClassSamples]

            arrayUnknownClassScores = dfGeneSetScores.reindex(listUnclassified).values.astype(float)
            structClassifier = KNeighborsClassifier(n_neighbors=30)
            structClassifier.fit(arrayKnownClassScores, arrayKnownClasses)

            arrayPredictedClass = structClassifier.predict(arrayUnknownClassScores)
            arrayPredictedClassProb = structClassifier.predict_proba(arrayUnknownClassScores)

            dfInferred = pd.DataFrame(data=np.zeros((len(listUnclassified),4), dtype=float),
                                      index=listUnclassified,
                                      columns=['InfClust','P(InfClust1)','P(InfClust2)','P(InfClust3)'])
            dfInferred['InfClust'] = arrayPredictedClass
            dfInferred[['P(InfClust1)','P(InfClust2)','P(InfClust3)']] = arrayPredictedClassProb

            dfInferred.to_pickle(pathTempFile)

            if flagPlotInfClust:

                listInfClust1 = [strSample for strSample in dfInferred[dfInferred['InfClust'] == 1].index.tolist()
                                 if strSample[-3:] == '01A']
                listInfClust2 = [strSample for strSample in dfInferred[dfInferred['InfClust'] == 2].index.tolist()
                                 if strSample[-3:] == '01A']
                listInfClust3 = [strSample for strSample in dfInferred[dfInferred['InfClust'] == 3].index.tolist()
                                 if strSample[-3:] == '01A']

                arrayGridSpec = matplotlib.gridspec.GridSpec(
                    nrows=1, ncols=3,
                    left=0.12, right=0.97,
                    bottom=0.14, top=0.95,
                    wspace=0.5)

                arrayColorNorm = matplotlib.colors.Normalize(vmin=0,
                                                             vmax=19)
                arrayColorsForMap = matplotlib.cm.ScalarMappable(norm=arrayColorNorm,
                                                                 cmap=matplotlib.cm.tab20)

                handFig = plt.figure()
                handFig.set_size_inches(w=8.5, h=3)

                handAx = plt.subplot(arrayGridSpec[0])
                handAx.scatter(dfGeneSetScores['iClust1 score'].reindex(listOfListsClusterPatients[0]).values.astype(float),
                               dfGeneSetScores['iClust2 score'].reindex(listOfListsClusterPatients[0]).values.astype(float),
                               c=arrayColorsForMap.to_rgba(0),
                               linewidths=0,
                               alpha=0.8,
                               s=10.0,
                               label='iClust1')
                handAx.scatter(dfGeneSetScores['iClust1 score'].reindex(listOfListsClusterPatients[1]).values.astype(float),
                               dfGeneSetScores['iClust2 score'].reindex(listOfListsClusterPatients[1]).values.astype(float),
                               c=arrayColorsForMap.to_rgba(2),
                               linewidths=0,
                               alpha=0.8,
                               s=10.0,
                               label='iClust2')
                handAx.scatter(dfGeneSetScores['iClust1 score'].reindex(listOfListsClusterPatients[2]).values.astype(float),
                               dfGeneSetScores['iClust2 score'].reindex(listOfListsClusterPatients[2]).values.astype(float),
                               c=arrayColorsForMap.to_rgba(4),
                               linewidths=0,
                               alpha=0.8,
                               s=10.0,
                               label='iClust3')

                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listInfClust1).values.astype(float),
                    dfGeneSetScores['iClust2 score'].reindex(listInfClust1).values.astype(float),
                    c=arrayColorsForMap.to_rgba(1),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust1 (inf.)')
                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listInfClust2).values.astype(float),
                    dfGeneSetScores['iClust2 score'].reindex(listInfClust2).values.astype(float),
                    c=arrayColorsForMap.to_rgba(3),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust2 (inf.)')
                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listInfClust3).values.astype(float),
                    dfGeneSetScores['iClust2 score'].reindex(listInfClust3).values.astype(float),
                    c=arrayColorsForMap.to_rgba(5),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust3 (inf.)')

                handAx.set_xlabel('iClust1 score')
                handAx.set_ylabel('iClust2 score')

                handAx = plt.subplot(arrayGridSpec[1])
                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listOfListsClusterPatients[0]).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listOfListsClusterPatients[0]).values.astype(float),
                    c=arrayColorsForMap.to_rgba(0),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust1')
                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listOfListsClusterPatients[1]).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listOfListsClusterPatients[1]).values.astype(float),
                    c=arrayColorsForMap.to_rgba(2),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust2')
                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listOfListsClusterPatients[2]).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listOfListsClusterPatients[2]).values.astype(float),
                    c=arrayColorsForMap.to_rgba(4),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust3')

                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listInfClust1).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listInfClust1).values.astype(float),
                    c=arrayColorsForMap.to_rgba(1),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust1 (inf.)')
                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listInfClust2).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listInfClust2).values.astype(float),
                    c=arrayColorsForMap.to_rgba(3),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust2 (inf.)')
                handAx.scatter(
                    dfGeneSetScores['iClust1 score'].reindex(listInfClust3).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listInfClust3).values.astype(float),
                    c=arrayColorsForMap.to_rgba(5),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust3 (inf.)')

                handAx.set_xlabel('iClust1 score')
                handAx.set_ylabel('iClust3 score')

                handAx = plt.subplot(arrayGridSpec[2])
                handAx.scatter(
                    dfGeneSetScores['iClust2 score'].reindex(listOfListsClusterPatients[0]).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listOfListsClusterPatients[0]).values.astype(float),
                    c=[arrayColorsForMap.to_rgba(0)]*len(listOfListsClusterPatients[0]),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust1')
                handAx.scatter(
                    dfGeneSetScores['iClust2 score'].reindex(listOfListsClusterPatients[1]).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listOfListsClusterPatients[1]).values.astype(float),
                    c=[arrayColorsForMap.to_rgba(2)]*len(listOfListsClusterPatients[1]),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust2')
                handAx.scatter(
                    dfGeneSetScores['iClust2 score'].reindex(listOfListsClusterPatients[2]).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listOfListsClusterPatients[2]).values.astype(float),
                    c=[arrayColorsForMap.to_rgba(4)]*len(listOfListsClusterPatients[2]),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust3')

                handAx.scatter(
                    dfGeneSetScores['iClust2 score'].reindex(listInfClust1).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listInfClust1).values.astype(float),
                    c=[arrayColorsForMap.to_rgba(1)]*len(listInfClust1),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust1 (inf.)')
                handAx.scatter(
                    dfGeneSetScores['iClust2 score'].reindex(listInfClust2).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listInfClust2).values.astype(float),
                    c=[arrayColorsForMap.to_rgba(3)]*len(listInfClust2),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust2 (inf.)')
                handAx.scatter(
                    dfGeneSetScores['iClust2 score'].reindex(listInfClust3).values.astype(float),
                    dfGeneSetScores['iClust3 score'].reindex(listInfClust3).values.astype(float),
                    c=[arrayColorsForMap.to_rgba(5)]*len(listInfClust3),
                    linewidths=0,
                    alpha=0.8,
                    s=10.0,
                    label='iClust3 (inf.)')

                handAx.set_xlabel('iClust2 score')
                handAx.set_ylabel('iClust3 score')

                for strFormat in Plot.listFigFormats:
                    handFig.savefig(os.path.join(PathDir.pathPlotFolder, 'InferredClusters.'+strFormat),
                                    dpi=300)

                plt.close(handFig)

        else:

            dfInferred = pd.read_pickle(pathTempFile)

        return dfInferred

    def int_clust_genesets(flagResult=False):

        listFiles = ['new.c1.vs.c23.tsv', 'new.c2.vs.c13.tsv', 'new.c3.vs.c12.tsv']
        listClusters = ['IntClust1', 'IntClust2', 'IntClust3']

        dictGeneSets = {}

        for iFile in range(len(listFiles)):
            strFile = listFiles[iFile]
            strClust = listClusters[iFile]

            dictGeneSets[strClust] = {}

            dfIn = pd.read_csv(os.path.join(PathDir.pathDataFolder, strFile),
                                 sep='\t', header=0, quotechar='"', quoting=2, index_col=0)

            listDownRegGenes = dfIn[dfIn['logFC'].values.astype(float) < 0.0].index.tolist()
            listUpRegGenes = dfIn[dfIn['logFC'].values.astype(float) > 0.0].index.tolist()

            dictGeneSets[strClust]['UpGenes']=listUpRegGenes
            dictGeneSets[strClust]['DownGenes']=listDownRegGenes

        return dictGeneSets

class PlotFunc:

    def sample_out_order(flagResult=False,
                         flagPerformExtraction=False):

        pathTempFile = os.path.join(PathDir.pathPlotFolder, 'TCGA-LIHC-outOrder.txt')
        pathAbundData = os.path.join(PathDir.pathPlotFolder, 'Fig1_TCGA-AbundData.csv')
        pathAbundZNormData = os.path.join(PathDir.pathPlotFolder, 'Fig1_TCGA-AbundDataZNorm.csv')

        if any([not os.path.exists(pathTempFile),
                not os.path.exists(pathAbundData),
                not os.path.exists(pathAbundZNormData)]):
            flagPerformExtraction = True

        if flagPerformExtraction:

            dfMetaLIHC = TCGAFunctions.lihc_metadata()
            dfInfClust = TCGAFunctions.infer_unclassified_clusters()

            dfRNA = TCGAFunctions.lihc_mess_rna()
            listRNAGenes = dfRNA.index.tolist()
            listRNASamples = dfRNA.columns.tolist()
            listRNASamplesClean = [strSample[0:len('TCGA-NN-NNNN-01A')] for strSample in listRNASamples]
            dfRNA.rename(columns=dict(zip(listRNASamples, listRNASamplesClean)), inplace=True)

            listOutGenes = Plot.listGenes.copy()
            listOutGenes.remove('MIMAT0000421')

            listOutRNAGenes = [strGene for strGene in listRNAGenes if strGene.split('|')[0] in listOutGenes]

            arrayRNAOutGenes = dfRNA.reindex(listOutRNAGenes).values.astype(float)
            arrayMeanVal = np.mean(arrayRNAOutGenes, axis=1)
            arrayStdDev = np.std(arrayRNAOutGenes, axis=1)

            dfRNAZNorm = pd.DataFrame(data=np.zeros((len(listOutRNAGenes), len(listRNASamplesClean)),
                                                    dtype=float),
                                      index=listOutRNAGenes,
                                      columns=listRNASamplesClean)
            for iGene in range(len(listOutRNAGenes)):
                dfRNAZNorm.loc[listOutRNAGenes[iGene],:] = \
                    (arrayRNAOutGenes[iGene,:] - arrayMeanVal[iGene])/arrayStdDev[iGene]

            dfMeth, dictOfDictsMethProbeData = TCGAFunctions.lihc_dna_me()
            listMethSamples = dfMeth.columns.tolist()
            listMethSamplesClean = [strSample[0:len('TCGA-NN-NNNN-01A')] for strSample in listMethSamples]
            dfMeth.rename(columns=dict(zip(listMethSamples, listMethSamplesClean)), inplace=True)

            listProbes = [Plot.dictGeneToProbe[strGene] for strGene in Plot.listGenes]
            arrayMethOutProbes = dfMeth.reindex(listProbes).values.astype(float)
            arrayMeanVal = np.mean(arrayMethOutProbes, axis=1)
            arrayStdDev = np.std(arrayMethOutProbes, axis=1)

            dfMethZNorm = pd.DataFrame(data=np.zeros((len(listProbes), len(listMethSamplesClean)),
                                                    dtype=float),
                                      index=listProbes,
                                      columns=listMethSamplesClean)
            for iProbe in range(len(listProbes)):
                dfMethZNorm.loc[listProbes[iProbe], :] = \
                    (arrayMethOutProbes[iProbe, :] - arrayMeanVal[iProbe]) / arrayStdDev[iProbe]


            dfMicRNA = TCGAFunctions.lihc_micro_rna()
            listMicRNASamples = dfMicRNA.columns.tolist()
            listMicRNASamplesClean = [strSample[0:len('TCGA-NN-NNNN-01A')] for strSample in listMicRNASamples]

            dfMicRNA.rename(columns=dict(zip(listMicRNASamples, listMicRNASamplesClean)), inplace=True)

            arrayMicRNAOut = dfMicRNA.reindex(['hsa-miR-122-5p']).values.astype(float)
            numMeanVal = np.mean(arrayMicRNAOut)
            numStdDev = np.std(arrayMicRNAOut)

            dfMicRNANorm = pd.DataFrame(data=np.zeros((1, len(listMicRNASamplesClean)),
                                                    dtype=float),
                                      index=['hsa-miR-122-5p'],
                                      columns=listMicRNASamplesClean)
            dfMicRNANorm.loc['hsa-miR-122-5p',:] = (arrayMicRNAOut-numMeanVal)/numStdDev



            # #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
            # # Extract the copy number data for these genes

            dfAllCopyNum = TCGAFunctions.lihc_copy_number()
            listCopyNumSamples = sorted(dfAllCopyNum['Sample'].unique().tolist())
            listCopyNumTumourSamples = [strSample for strSample in listCopyNumSamples if strSample[13:15]=='01']
            listCopyNumTumourSamplesClean = [strSample[0:len('TCGA-NN-NNNN-NNA')] for strSample in listCopyNumTumourSamples]

            dictProbeToChr = dictOfDictsMethProbeData['Chr']
            dictProbeToGenLoc = dictOfDictsMethProbeData['GenCoord']

            listOutGenesClean = ['CNV:'+strGene for strGene in listOutGenes + ['MIMAT0000421']]
            dfCopyNumClean = pd.DataFrame(data=np.zeros((len(listOutGenesClean), len(listCopyNumTumourSamples)),
                                                        dtype=float),
                                          index=listOutGenesClean,
                                          columns=listCopyNumTumourSamples)
            for strGene in listOutGenesClean:
                strProbe = Plot.dictGeneToProbe[strGene.split('CNV:')[1]]
                if np.bitwise_and(not dictProbeToChr[strProbe] == 'X',
                                  not dictProbeToChr[strProbe] == 'Y'):
                    numChrom = int(dictProbeToChr[strProbe])
                    numMinGenomicLoc = np.min(dictProbeToGenLoc[strProbe])
                    numMaxGenomicLoc = np.max(dictProbeToGenLoc[strProbe])

                    arrayCNAInRange = np.bitwise_and(dfAllCopyNum['Start'] < numMinGenomicLoc,
                                                     dfAllCopyNum['End'] > numMaxGenomicLoc)
                    arrayIsCNARowOfInt = np.bitwise_and(dfAllCopyNum['Chromosome'] == str(numChrom),
                                                       arrayCNAInRange)
                    arrayCNARowsOfInt = np.where(arrayIsCNARowOfInt)[0]

                    for iRow in range(len(arrayCNARowsOfInt)):
                        strSample = dfAllCopyNum['Sample'].iloc[arrayCNARowsOfInt[iRow]]
                        if strSample in listCopyNumTumourSamples:
                            dfCopyNumClean.loc[strGene, strSample] = dfAllCopyNum['Segment_Mean'].iloc[arrayCNARowsOfInt[iRow]]

            dfCopyNumClean.rename(columns=dict(zip(listCopyNumTumourSamples,listCopyNumTumourSamplesClean)), inplace=True)

            # #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
            # # Combine

            setCommonSamples = set(listMicRNASamplesClean).intersection(
                set(listMethSamplesClean).intersection(set(listRNASamplesClean)))

            setCommonTumourSamples = set([strSample for strSample in setCommonSamples if strSample[13:15]=='01']).intersection(set(listCopyNumTumourSamplesClean))
            listCommonNormalSamples = [strSample for strSample in setCommonSamples if strSample[13:15]=='11']

            listTumourSamplesOrdered = []
            for iClust in range(3):
                listKnownClusterPatients = dfMetaLIHC[dfMetaLIHC['iCluster'] == iClust + 1].index.tolist()
                listKnownClusterSamples = [strPatient + '-01A' for strPatient in listKnownClusterPatients]

                listInfClusterSamples = dfInfClust[dfInfClust['InfClust'] == iClust + 1].index.tolist()

                listClusterSamples = list(
                    set(listKnownClusterSamples+listInfClusterSamples).intersection(setCommonTumourSamples))

                dfToCluster = pd.concat([dfMicRNANorm[listClusterSamples],
                                         dfRNAZNorm[listClusterSamples],
                                         dfMethZNorm[listClusterSamples]],
                                       axis=0)

                arrayAbundLink = \
                    SciPyClus.linkage(
                        np.nan_to_num(dfToCluster.transpose().values.astype(float)),
                        method='ward',
                        optimal_ordering=True)
                arrayAbundGeneOrder = SciPyClus.leaves_list(arrayAbundLink)

                listTumourSamplesOrdered += [listClusterSamples[i] for i in arrayAbundGeneOrder]


            dfToCluster = pd.concat([dfMicRNANorm[listCommonNormalSamples],
                                     dfRNAZNorm[listCommonNormalSamples],
                                     dfMethZNorm[listCommonNormalSamples]],
                                    axis=0)

            arrayAbundLink = \
                SciPyClus.linkage(
                    np.nan_to_num(dfToCluster.transpose().values.astype(float)),
                    method='ward',
                    optimal_ordering=True)
            arrayAbundGeneOrder = SciPyClus.leaves_list(arrayAbundLink)

            listNormalSamplesOrdered = [listCommonNormalSamples[i] for i in arrayAbundGeneOrder]

            dfClustOrderOut = pd.DataFrame({
                'Sample':listTumourSamplesOrdered+listNormalSamplesOrdered,
                'Group':['Tumour']*len(listTumourSamplesOrdered) + ['Normal']*len(listNormalSamplesOrdered)})
            dfClustOrderOut.to_csv(pathTempFile)

            dfTargetData = pd.concat([dfRNA.reindex(listOutRNAGenes).transpose(),
                                      dfMicRNA.reindex(['hsa-miR-122-5p']).transpose(),
                                      dfMeth.reindex(listProbes).transpose(),
                                      dfCopyNumClean.transpose()],
                                     axis=1)
            dfTargetData.reindex(listTumourSamplesOrdered+listNormalSamplesOrdered).to_csv(
                os.path.join(pathAbundData), sep=',')

            dfZNormData = pd.concat([dfRNAZNorm.transpose(),
                                     dfMicRNANorm.transpose(),
                                     dfMethZNorm.transpose()],
                                    axis=1)
            dfZNormData.reindex(listTumourSamplesOrdered+listNormalSamplesOrdered).to_csv(
                os.path.join(pathAbundZNormData), sep=',')

        else:
            dfClustOrderOut = pd.read_csv(pathTempFile, header=0, index_col=None)
            dfTargetData = pd.read_csv(pathAbundData, header=0, index_col=0)
            dfZNormData = pd.read_csv(pathAbundZNormData, header=0, index_col=0)

            listTumourSamplesOrdered = dfClustOrderOut['Sample'][dfClustOrderOut['Group']=='Tumour'].values.tolist()
            listNormalSamplesOrdered = dfClustOrderOut['Sample'][dfClustOrderOut['Group']=='Normal'].values.tolist()

        return {'Tumour': listTumourSamplesOrdered,
                'Normal': listNormalSamplesOrdered,
                'AbundData':dfTargetData,
                'ZNormData':dfZNormData}

    def rna_abund(flagResult=False,
                  handAxTumour='undefined',
                  handAxNormal='undefined',
                  handAxRNACMap='undefined'):

        dictPlotOrder = PlotFunc.sample_out_order()
        listOutTumourSamples = dictPlotOrder['Tumour']
        listOutNormalSamples = dictPlotOrder['Normal']

        dfTargetDataZNorm = dictPlotOrder['ZNormData']
        # listOutCol = [strCol for strCol in dfTargetDataZNorm.columns.tolist()
        #               if not np.bitwise_or(strCol.startswith('cg'), strCol.startswith('CNV:'))]

        listOutCol = []
        for strCol in Plot.listGenes:
            if strCol.startswith('MIMAT'):
                strColToMatch = Plot.dictMicRNAIDToName[strCol]
            else:
                strColToMatch = strCol + '|'
            strColMatched = [strColMatch for strColMatch in dfTargetDataZNorm.columns.tolist()
                             if strColMatch.startswith(strColToMatch)][0]
            listOutCol.append(strColMatched)

        handTumour = handAxTumour.matshow(
            dfTargetDataZNorm[listOutCol].reindex(listOutTumourSamples),
            vmin=-3.5, vmax=3.5,
            cmap=plt.cm.PRGn, aspect='auto')

        handAxTumour.set_xticks([])
        handAxTumour.set_yticks([])
        for iGene in range(len(Plot.listGenes)):
            strGene = Plot.listGenes[iGene]
            if strGene.startswith('MIMAT'):
                strGeneOut = Plot.dictMicRNAIDToName[strGene]
            else:
                strGeneOut = strGene

            handAxTumour.text(iGene, -1.5,
                        strGeneOut,
                        ha='center', va='bottom',
                        fontsize=Plot.numFontSize*0.75,
                              rotation=90,
                        fontstyle='italic')

            if iGene < len(Plot.listGenes)-1:
                handAxTumour.axvline(x=iGene+0.5,
                               ymin=0.0, ymax=1.0,
                               color='0.5', lw=0.25)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxTumour.spines[strAxLoc].set_linewidth(0.1)

        handNormal = handAxNormal.matshow(
            dfTargetDataZNorm[listOutCol].reindex(listOutNormalSamples),
            vmin=-3.5, vmax=3.5,
            cmap=plt.cm.PRGn, aspect='auto')

        handAxNormal.set_xticks([])
        handAxNormal.set_yticks([])
        for iGene in range(len(Plot.listGenes)):
            if iGene < len(Plot.listGenes)-1:
                handAxNormal.axvline(x=iGene+0.5,
                                     ymin=0.0, ymax=1.0,
                                     color='0.5', lw=0.25)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxNormal.spines[strAxLoc].set_linewidth(0.1)

        handSigColorBar = plt.colorbar(handTumour, cax=handAxRNACMap,
                                       orientation='horizontal',
                                       extend='both')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxRNACMap.spines[strAxLoc].set_linewidth(0.1)

        return flagResult

    def meth_abund(flagResult=False,
                   handAxTumour='undefined',
                   handAxNormal='undefined',
                   handAxDNAmeCMap='undefined'):

        dictPlotOrder = PlotFunc.sample_out_order()
        listOutTumourSamples = dictPlotOrder['Tumour']
        listOutNormalSamples = dictPlotOrder['Normal']

        dfTargetDataZNorm = dictPlotOrder['ZNormData']

        listOutGenesClean = []
        listOutProbes = []
        for strGene in Plot.listGenes:
            if strGene.startswith('MIMAT'):
                listOutGenesClean.append(Plot.dictMicRNAIDToName[strGene])
            else:
                listOutGenesClean.append(strGene)
            listOutProbes.append(Plot.dictGeneToProbe[strGene])

        handTumour = handAxTumour.matshow(
            dfTargetDataZNorm[listOutProbes].reindex(listOutTumourSamples),
            vmin=-3.5, vmax=3.5,
            cmap=plt.cm.BrBG, aspect='auto')

        handAxTumour.set_xticks([])
        handAxTumour.set_yticks([])
        for iGene in range(len(listOutProbes)):
            strProbe = listOutProbes[iGene]

            handAxTumour.text(iGene, -1.5,
                              strProbe,
                              ha='center', va='bottom',
                              fontsize=Plot.numFontSize*0.75,
                              rotation=90,
                              fontstyle='italic')

            if iGene < len(Plot.listGenes)-1:
                handAxTumour.axvline(x=iGene+0.5,
                                     ymin=0.0, ymax=1.0,
                                     color='0.5', lw=0.25)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxTumour.spines[strAxLoc].set_linewidth(0.1)

        handNormal = handAxNormal.matshow(
            dfTargetDataZNorm[listOutProbes].reindex(listOutNormalSamples),
            vmin=-3.5, vmax=3.5,
            cmap=plt.cm.BrBG, aspect='auto')

        handAxNormal.set_xticks([])
        handAxNormal.set_yticks([])
        for iGene in range(len(Plot.listGenes)):
            if iGene < len(Plot.listGenes)-1:
                handAxNormal.axvline(x=iGene+0.5,
                                     ymin=0.0, ymax=1.0,
                                     color='0.5', lw=0.25)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxNormal.spines[strAxLoc].set_linewidth(0.1)

        handSigColorBar = plt.colorbar(handTumour, cax=handAxDNAmeCMap,
                                       orientation='horizontal',
                                       extend='both')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxDNAmeCMap.spines[strAxLoc].set_linewidth(0.1)

        return flagResult

    def copy_number(flagResult=False,
                    handAxTumour='undefined',
                    handAxCNVCMap='undefined'):

        dictPlotOrder = PlotFunc.sample_out_order()
        listOutTumourSamples = dictPlotOrder['Tumour']

        dfTargetData = dictPlotOrder['AbundData']

        # listOutCNVCols = [strCol for strCol in dfTargetData.columns.tolist() if strCol.startswith('CNV:')]
        listOutCNVCols = ['CNV:'+strGene for strGene in Plot.listGenes if 'CNV:'+strGene in dfTargetData.columns.tolist()]

        handTumour = handAxTumour.matshow(
            dfTargetData[listOutCNVCols].reindex(listOutTumourSamples),
            vmin=-2, vmax=2,
            cmap=plt.cm.bwr, aspect='auto')

        handAxTumour.set_xticks([])
        handAxTumour.set_yticks([])
        for iGene in range(len(listOutCNVCols)):
            strGene = listOutCNVCols[iGene].split('CNV:')[1]
            if strGene.startswith('MIMAT'):
                strGeneToDisp = Plot.dictMicRNAIDToName[strGene]
            else:
                strGeneToDisp = strGene

            handAxTumour.text(iGene, -1.5,
                              strGeneToDisp,
                              ha='center', va='bottom',
                              fontsize=Plot.numFontSize*0.75,
                              rotation=90,
                              fontstyle='italic')

            if iGene < len(Plot.listGenes)-1:
                handAxTumour.axvline(x=iGene+0.5,
                                     ymin=0.0, ymax=1.0,
                                     color='0.5', lw=0.25)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxTumour.spines[strAxLoc].set_linewidth(0.1)

        handSigColorBar = plt.colorbar(handTumour, cax=handAxCNVCMap,
                                       orientation='horizontal',
                                       extend='both')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()


        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxCNVCMap.spines[strAxLoc].set_linewidth(0.1)

        return flagResult

    def iclust_heatmap(flagResult=False,
                       handAxTumour='undefined'):

        dfMetaLIHC = TCGAFunctions.lihc_metadata()
        dfInfClust = TCGAFunctions.infer_unclassified_clusters()

        dictPlotData = PlotFunc.sample_out_order()
        listOutTumourSamples = dictPlotData['Tumour']

        listClust = []
        listClustSamples = []
        listClustColour = []
        numBaseVal = 0
        for iClust in range(3):

            listKnownClusterPatients = dfMetaLIHC[dfMetaLIHC['iCluster'] == iClust + 1].index.tolist()
            listKnownClusterSamples = [strPatient + '-01A' for strPatient in listKnownClusterPatients]

            listInfClusterSamples = dfInfClust[dfInfClust['InfClust'] == iClust + 1].index.tolist()

            listKnownTumourCluster = list(
                set(listKnownClusterSamples).intersection(set(listOutTumourSamples)))
            listInfTumourCluster = list(
                set(listInfClusterSamples).intersection(set(listOutTumourSamples)))

            listForCluster = listKnownTumourCluster + listInfTumourCluster
            listClustSamples += listForCluster

            listClustColour += [numBaseVal]*len(listKnownTumourCluster) + [numBaseVal+1]*len(listInfTumourCluster)
            listClust += [f'iClust{iClust+1}']*len(listKnownTumourCluster) + \
                         [f'iClust{iClust+1}-inf']*len(listInfTumourCluster)

            numBaseVal += 2

        dfClusters = pd.DataFrame({'Cluster':listClust, 'Color':listClustColour}, index=listClustSamples)

        handAxTumour.matshow(dfClusters[['Color']].reindex(listOutTumourSamples),
                             cmap=plt.cm.tab20,
                             vmin=0, vmax=19,
                             aspect='auto')
        handAxTumour.set_xticks([])
        handAxTumour.set_yticks([])

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxTumour.spines[strAxLoc].set_linewidth(0.1)

        listClusters = dfClusters['Cluster'].unique().tolist()

        arrayXRange = handAxTumour.get_xlim()
        arrayYRange = handAxTumour.get_ylim()

        # arrayColorNorm = matplotlib.colors.Normalize(vmin=0, vmax=19)
        # arrayColorMap = matplotlib.cm.ScalarMappable(norm=arrayColorNorm, cmap=plt.cm.tab20)

        for iCluster in range(len(listClusters)):
            handAxTumour.scatter(arrayXRange[0]-np.ptp(arrayXRange),
                                 arrayYRange[0]-np.ptp(arrayYRange),
                                 color=Plot.arrayColorMap.to_rgba(iCluster),
                                 label=listClusters[iCluster])

        handAxTumour.scatter(arrayXRange[0]-np.ptp(arrayXRange),
                             arrayYRange[0]-np.ptp(arrayYRange),
                             color='k',
                             label='Adjacent normal')
        handAxTumour.set_xlim(arrayXRange)
        handAxTumour.set_ylim(arrayYRange)

        plt.legend(loc='upper right',
                   bbox_to_anchor=(1.00, -0.01),
                   fontsize=Plot.numFontSize*0.75,
                   scatterpoints=1,
                   ncol=1,
                   fancybox=True,
                   shadow=True)

        return flagResult

    def scatter_plots(flagResult=False,
                      dictAxPos=dict()):

        dfMetaLIHC = TCGAFunctions.lihc_metadata()
        dfInfClust = TCGAFunctions.infer_unclassified_clusters()

        dictPlotData = PlotFunc.sample_out_order()
        listOutTumourSamples = dictPlotData['Tumour']
        listOutNormalSamples = dictPlotData['Normal']

        listClust = []
        listClustSamples = []
        listClustColour = []
        numBaseVal = 0
        for iClust in range(3):

            listKnownClusterPatients = dfMetaLIHC[dfMetaLIHC['iCluster'] == iClust + 1].index.tolist()
            listKnownClusterSamples = [strPatient + '-01A' for strPatient in listKnownClusterPatients]

            listInfClusterSamples = dfInfClust[dfInfClust['InfClust'] == iClust + 1].index.tolist()

            listKnownTumourCluster = list(
                set(listKnownClusterSamples).intersection(set(listOutTumourSamples)))
            listInfTumourCluster = list(
                set(listInfClusterSamples).intersection(set(listOutTumourSamples)))

            listForCluster = listKnownTumourCluster + listInfTumourCluster
            listClustSamples += listForCluster

            listClustColour += [numBaseVal]*len(listKnownTumourCluster) + [numBaseVal+1]*len(listInfTumourCluster)
            listClust += [f'iClust{iClust+1}']*len(listKnownTumourCluster) + \
                         [f'iClust{iClust+1}-inf']*len(listInfTumourCluster)

            numBaseVal += 2

        dfClusters = pd.DataFrame({'Cluster':listClust, 'Color':listClustColour}, index=listClustSamples)

        dfTargetData = dictPlotData['AbundData']
        listDataCol = dfTargetData.columns.tolist()

        numScatterCols = 4
        iRow = 0
        iCol = 0
        for strGene in Plot.listGenes:
            listRNACols = [strCol for strCol in listDataCol
                           if strCol.startswith(strGene+'|')]
            if len(listRNACols) == 1:
                strRNACol = listRNACols[0]
            else:
                if strGene in Plot.dictMicRNAIDToName.keys():
                    strRNACol = Plot.dictMicRNAIDToName[strGene]

            strProbe = Plot.dictGeneToProbe[strGene]
            if strGene.startswith('MIMAT'):
                strGeneOut = Plot.dictMicRNAIDToName[strGene]
            else:
                strGeneOut = strGene

            for iClust in range(6):
                listClusterSamples = dfClusters[dfClusters['Color']==iClust+1].index.tolist()
                dictAxPos[strGene].scatter(dfTargetData[strProbe].reindex(listClusterSamples),
                    dfTargetData[strRNACol].reindex(listClusterSamples), alpha=0.6,
                                           zorder=7, s=3, lw=0.0,
                                           c=[Plot.arrayColorMap.to_rgba(iClust)]*len(listClusterSamples))
            dictAxPos[strGene].scatter(dfTargetData[strProbe].reindex(listOutNormalSamples),
                                       dfTargetData[strRNACol].reindex(listOutNormalSamples),
                                       color='k', alpha=0.7,
                                       zorder=6, s=5, lw=0.0)
            dictAxPos[strGene].set_title(strGeneOut, fontsize=6)

            dictAxPos[strGene].set_xticks([0, 0.5, 1.0])
            dictAxPos[strGene].set_xlim([-0.02, 1.02])

            dictAxPos[strGene].spines['top'].set_visible(False)
            dictAxPos[strGene].spines['right'].set_visible(False)

            arrayYLim = dictAxPos[strGene].get_ylim()
            dictAxPos[strGene].set_ylim([-0.2, arrayYLim[1]])
            arrayTickLoc = plt.MaxNLocator(3)

            dictAxPos[strGene].yaxis.set_major_locator(arrayTickLoc)
            for handTick in dictAxPos[strGene].yaxis.get_major_ticks():
                handTick.label.set_fontsize(6)
            for handTick in dictAxPos[strGene].xaxis.get_major_ticks():
                handTick.label.set_fontsize(6)

            dictAxPos[strGene].set_ylabel('')

            if iRow == 2:
                dictAxPos[strGene].set_xlabel(r'${\beta}$-value', fontsize=6)
                dictAxPos[strGene].set_xticklabels(['0.0', '0.5', '1.0'])
            else:
                dictAxPos[strGene].set_xticklabels([])

            if iCol >= numScatterCols-1:
                iRow += 1
                iCol = 0
            else:
                iCol += 1

        return flagResult

class Plot:

    dictMicRNANameToID = {'hsa-miR-122-5p':'MIMAT0000421'}
    dictMicRNAIDToName = dict(zip(dictMicRNANameToID.values(), dictMicRNANameToID.keys()))

    listFigFormats = ['png', 'pdf']
    numFontSize = 6

    arrayColorNorm = matplotlib.colors.Normalize(vmin=0, vmax=19)
    arrayColorMap = matplotlib.cm.ScalarMappable(norm=arrayColorNorm, cmap=plt.cm.tab20)

    listGenes = [
        'BCO2',
        'CDKN2A',
        'CPS1',
        'HHIP',
        dictMicRNANameToID['hsa-miR-122-5p'],
        'MT1E',
        'MT1M',
        'PSAT1',
        'PTGR1',
        'PZP',
        'TMEM106A',
        'TTC36']


    dictGeneToProbe = {'BCO2':'cg26581504',
                       'CPS1':'cg21967368', # same as TCGA paper
                       'CDKN2A':'cg13601799',
                       'TTC36':'cg24222440',
                       dictMicRNANameToID['hsa-miR-122-5p']:'cg00481280',
                       'HHIP':'cg26621699', # cg23109129 in TCGA paper
                       'PTGR1':'cg13397260', # cg13831329 in TCGA paper
                       'TMEM106A':'cg21211480', # same as TCGA paper
                       'MT1E':'cg04793813', # cg02512505 in TCGA paper
                       'PZP':'cg08729600',
                       'MT1M':'cg10638827', # cg15134649 in TCGA paper
                       'PSAT1':'cg13740985'}


    def figure_one(flagResult=False):

        tupleFigSize = (6.5, 9.5)

        numCMapHeight = 0.0075

        dictPlotOrder = PlotFunc.sample_out_order()
        listOutTumourSamples = dictPlotOrder['Tumour']
        listOutNormalSamples = dictPlotOrder['Normal']

        dfMetaLIHC = TCGAFunctions.lihc_metadata()
        dfInfClust = TCGAFunctions.infer_unclassified_clusters()

        listClust = []
        listClustSamples = []
        listClustColour = []
        numBaseVal = 0
        for iClust in range(3):

            listKnownClusterPatients = dfMetaLIHC[dfMetaLIHC['iCluster'] == iClust + 1].index.tolist()
            listKnownClusterSamples = [strPatient + '-01A' for strPatient in listKnownClusterPatients]

            listInfClusterSamples = dfInfClust[dfInfClust['InfClust'] == iClust + 1].index.tolist()

            listKnownTumourCluster = list(
                set(listKnownClusterSamples).intersection(set(listOutTumourSamples)))
            listInfTumourCluster = list(
                set(listInfClusterSamples).intersection(set(listOutTumourSamples)))

            listForCluster = listKnownTumourCluster + listInfTumourCluster
            listClustSamples += listForCluster

            listClustColour += [numBaseVal]*len(listKnownTumourCluster) + [numBaseVal+1]*len(listInfTumourCluster)
            listClust += [f'iClust{iClust+1}']*len(listKnownTumourCluster) + \
                         [f'iClust{iClust+1}-inf']*len(listInfTumourCluster)

            numBaseVal += 2

        dfClusters = pd.DataFrame({'Cluster':listClust, 'Color':listClustColour}, index=listClustSamples)
        dfClusters['Cluster'].to_csv(os.path.join(PathDir.pathPlotFolder, 'Sample-iClust.csv'))

        arrayHorizLinePos = np.array([np.sum((dfClusters['Color']<=1).values.astype(bool)),
                                      np.sum((dfClusters['Color']<=3).values.astype(bool))])

        numTumourHeatMapPanelHeight = 0.40
        numHeightPerSample = numTumourHeatMapPanelHeight/len(listOutTumourSamples)
        numNormalHeatMapPanelHeight = numHeightPerSample*len(listOutNormalSamples)

        numTopHeatMapPanelBaseY = 0.53
        numHeatMapCMapBaseY = 0.45
        numPanelWidth = 0.26
        numXPanelSpacer = 0.02
        numPanelBaseX = 0.07
        numIntClustPanelWidth = 0.05

        dictPanelLoc = {'HeatMap:Tumour_RNA':[numPanelBaseX, numTopHeatMapPanelBaseY,
                                              numPanelWidth, numTumourHeatMapPanelHeight],
                        'HeatMap:Normal_RNA':[numPanelBaseX, numTopHeatMapPanelBaseY-(numNormalHeatMapPanelHeight+0.01),
                                              numPanelWidth, numNormalHeatMapPanelHeight],
                        'HeatMap_cmap:RNA':[numPanelBaseX+(numPanelWidth/4), numHeatMapCMapBaseY,
                                            numPanelWidth/2, numCMapHeight],
                        'HeatMap:Tumour_DNAme':[numPanelBaseX+numXPanelSpacer+numPanelWidth, numTopHeatMapPanelBaseY,
                                                numPanelWidth, numTumourHeatMapPanelHeight],
                        'HeatMap:Normal_DNAme':[numPanelBaseX+numXPanelSpacer+numPanelWidth, numTopHeatMapPanelBaseY-(numNormalHeatMapPanelHeight+0.01),
                                                numPanelWidth, numNormalHeatMapPanelHeight],
                        'HeatMap_cmap:DNAme':[numPanelBaseX+numXPanelSpacer+numPanelWidth+(numPanelWidth/4), numHeatMapCMapBaseY,
                                              numPanelWidth/2, numCMapHeight],
                        'HeatMap:Tumour_CNV':[numPanelBaseX+(numXPanelSpacer+numPanelWidth)*2,
                                              numTopHeatMapPanelBaseY,
                                              numPanelWidth, numTumourHeatMapPanelHeight],
                        'HeatMap_cmap:CNV':[numPanelBaseX+2*(numXPanelSpacer+numPanelWidth)+(numPanelWidth/4),
                                            numHeatMapCMapBaseY,
                                            numPanelWidth/2, numCMapHeight],
                        'HeatMap:Tumour_iClust':[numPanelBaseX+(numXPanelSpacer+numPanelWidth)*3,
                                                 numTopHeatMapPanelBaseY,
                                                 numIntClustPanelWidth, numTumourHeatMapPanelHeight]}

        numRHS = numPanelBaseX+(numXPanelSpacer+numPanelWidth)*3 + numIntClustPanelWidth

        numScatterTop = 0.37

        numScatterCols = 4
        arrayGridSpec = matplotlib.gridspec.GridSpec(nrows=3, ncols=numScatterCols,
                                                     bottom=0.05, top=numScatterTop,
                                                     left=numPanelBaseX, right=numRHS,
                                                     hspace=0.6, wspace=0.6)

        handFig = plt.figure(figsize=tupleFigSize)

        # # # # # #       #       #       #       #       #       #       #
        # RNA-seq

        handAxTumour = handFig.add_axes(dictPanelLoc['HeatMap:Tumour_RNA'])
        handAxNormal = handFig.add_axes(dictPanelLoc['HeatMap:Normal_RNA'])
        handAxRNACMap = handFig.add_axes(dictPanelLoc['HeatMap_cmap:RNA'])

        _ = PlotFunc.rna_abund(handAxTumour=handAxTumour,
                               handAxNormal=handAxNormal,
                               handAxRNACMap=handAxRNACMap)

        for iLinePos in range(len(arrayHorizLinePos)):
            handAxTumour.axhline(y=arrayHorizLinePos[iLinePos],
                                 xmin=0, xmax=1, color='k', lw=0.75)

        structAxPos = handAxRNACMap.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     structAxPos.y0-2.5*structAxPos.height, 'RNA abundance\n($z$-score)',
                     ha='center', va='top',
                     fontsize=Plot.numFontSize*0.7)

        structAxPos = handAxTumour.get_position()
        handFig.text(structAxPos.x0-0.1*structAxPos.width,
                     structAxPos.y0+0.5*structAxPos.height,
                     'Tumour',
                     ha='right', va='center',
                     rotation=90,
                     fontsize=Plot.numFontSize,
                     fontweight='bold')

        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     0.999,
                     'RNA abundance',
                     ha='center', va='top',
                     fontsize=Plot.numFontSize,
                     fontweight='bold')

        structAxPos = handAxNormal.get_position()
        handFig.text(structAxPos.x0-0.1*structAxPos.width,
                     structAxPos.y0+0.5*structAxPos.height,
                     'Adjacent\nnormal',
                     ha='center', va='center',
                     rotation=90,
                     fontsize=Plot.numFontSize,
                     fontweight='bold')


        #       #       #       #       #       #       #       #       #
        # DNAme
        handAxTumour = handFig.add_axes(dictPanelLoc['HeatMap:Tumour_DNAme'])
        handAxNormal = handFig.add_axes(dictPanelLoc['HeatMap:Normal_DNAme'])
        handAxDNAmeCMap = handFig.add_axes(dictPanelLoc['HeatMap_cmap:DNAme'])

        _ = PlotFunc.meth_abund(handAxTumour=handAxTumour,
                               handAxNormal=handAxNormal,
                                handAxDNAmeCMap=handAxDNAmeCMap)

        for iLinePos in range(len(arrayHorizLinePos)):
            handAxTumour.axhline(y=arrayHorizLinePos[iLinePos],
                                 xmin=0, xmax=1, color='k', lw=0.75)

        structAxPos = handAxDNAmeCMap.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     structAxPos.y0-2.5*structAxPos.height, 'Probe methylation\n($z$-score)',
                     ha='center', va='top',
                     fontsize=Plot.numFontSize*0.7)

        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     0.999,
                     'DNA methylation',
                     ha='center', va='top',
                     fontsize=Plot.numFontSize,
                     fontweight='bold')

        #       #       #       #       #       #       #       #       #
        # CNV
        handAxTumour = handFig.add_axes(dictPanelLoc['HeatMap:Tumour_CNV'])
        handAxCNVCMap = handFig.add_axes(dictPanelLoc['HeatMap_cmap:CNV'])

        _ = PlotFunc.copy_number(handAxTumour=handAxTumour,
                                 handAxCNVCMap=handAxCNVCMap)

        for iLinePos in range(len(arrayHorizLinePos)):
            handAxTumour.axhline(y=arrayHorizLinePos[iLinePos],
                                 xmin=0, xmax=1, color='k', lw=0.75)

        structAxPos = handAxCNVCMap.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     structAxPos.y0-2.5*structAxPos.height, 'Copy number variation',
                     ha='center', va='top',
                     fontsize=Plot.numFontSize*0.7)

        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     0.999,
                     'Gene amplification/loss',
                     ha='center', va='top',
                     fontsize=Plot.numFontSize,
                     fontweight='bold')

        # # # # # # # # #       #       #       #       #       #       #
        # Integrated-data cluster (iCluster)
        handAxTumour = handFig.add_axes(dictPanelLoc['HeatMap:Tumour_iClust'])
        _ = PlotFunc.iclust_heatmap(handAxTumour=handAxTumour)

        for iLinePos in range(len(arrayHorizLinePos)):
            handAxTumour.axhline(y=arrayHorizLinePos[iLinePos],
                                 xmin=0, xmax=1, color='k', lw=0.75)

        structAxPos = handAxTumour.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     0.999,
                     'Subtype',
                     ha='center', va='top',
                     fontsize=Plot.numFontSize,
                     fontweight='bold')

        # # # # # # # # #       #       #       #       #       #       #
        # Scatter plots

        dictScatterPlotPos = dict()
        iRow = 0
        iCol = 0
        for strGene in Plot.listGenes:
            handAx = plt.subplot(arrayGridSpec[iRow, iCol])
            dictScatterPlotPos[strGene] = handAx
            if iCol >= numScatterCols-1:
                iRow += 1
                iCol = 0
            else:
                iCol += 1

        _ = PlotFunc.scatter_plots(dictAxPos=dictScatterPlotPos)

        iRow = 0
        iCol = 0
        for strGene in Plot.listGenes:
            if iCol == 0:
                structAxPos = dictScatterPlotPos[strGene].get_position()
                handFig.text(0.025,
                             structAxPos.y0 + structAxPos.height*0.5,
                             'Transcript\nabundance',
                             ha='center', va='center',
                             rotation=90,
                             fontsize=Plot.numFontSize)

            if iCol >= numScatterCols-1:
                iRow += 1
                iCol = 0
            else:
                iCol += 1


        handFig.text(0.01, 0.999, '(A)',
                     fontsize=10, fontweight='bold',
                     ha='left', va='top')
        handFig.text(0.01, numScatterTop+0.02, '(B)',
                     fontsize=10, fontweight='bold',
                     ha='left', va='top')

        for strFormat in Plot.listFigFormats:
            handFig.savefig(os.path.join(PathDir.pathPlotFolder, 'Figure1.'+strFormat),
                            dpi=300)
        plt.close(handFig)

        return flagResult


    def supp_fig_one(flagResult=False):

        numCNVDelThresh = -0.5

        listTargetedGenes = Plot.listGenes
        numTargets = len(listTargetedGenes)

        dictPlotOrder = PlotFunc.sample_out_order()
        listOutTumourSamples = dictPlotOrder['Tumour']
        listOutNormalSamples = dictPlotOrder['Normal']

        dfMetaLIHC = TCGAFunctions.lihc_metadata()
        dfInfClust = TCGAFunctions.infer_unclassified_clusters()

        listClust = []
        listClustSamples = []
        listClustColour = []
        numBaseVal = 0
        for iClust in range(3):

            listKnownClusterPatients = dfMetaLIHC[dfMetaLIHC['iCluster'] == iClust + 1].index.tolist()
            listKnownClusterSamples = [strPatient + '-01A' for strPatient in listKnownClusterPatients]

            listInfClusterSamples = dfInfClust[dfInfClust['InfClust'] == iClust + 1].index.tolist()

            listKnownTumourCluster = list(
                set(listKnownClusterSamples).intersection(set(listOutTumourSamples)))
            listInfTumourCluster = list(
                set(listInfClusterSamples).intersection(set(listOutTumourSamples)))

            listForCluster = listKnownTumourCluster + listInfTumourCluster
            listClustSamples += listForCluster

            listClustColour += [numBaseVal]*len(listKnownTumourCluster) + [numBaseVal+1]*len(listInfTumourCluster)
            listClust += [f'iClust{iClust+1}']*len(listKnownTumourCluster) + \
                         [f'iClust{iClust+1}-inf']*len(listInfTumourCluster)

            numBaseVal += 2

        dfClusters = pd.DataFrame({'Cluster':listClust,
                                   'Color':listClustColour},
                                  index=listClustSamples)

        numPatients = len(listOutTumourSamples)

        dfAbundData = dictPlotOrder['AbundData']
        listDataCols = dfAbundData.columns.tolist()
        listCNVCols = [strCol for strCol in listDataCols if strCol.startswith('CNV:')]

        listGenes = Plot.listGenes
        listProbes = [Plot.dictGeneToProbe[strGene] for strGene in listGenes]

        dfIsDel = dfAbundData[listCNVCols].reindex(listOutTumourSamples) < numCNVDelThresh
        dfIsHyperMeth = pd.DataFrame(data=np.zeros((numPatients, numTargets), dtype=bool),
                                     index=listOutTumourSamples,
                                     columns=listProbes)
        for strProbe in listProbes:
            sliceNorm = dfAbundData[strProbe].reindex(listOutNormalSamples)
            numThresh = np.max(sliceNorm.values.astype(float))
            dfIsHyperMeth.loc[listOutTumourSamples,strProbe] = \
                dfAbundData[strProbe].reindex(listOutTumourSamples).values.astype(float) > numThresh

        arrayIsHyperMethNoDel = np.bitwise_and(
            dfIsHyperMeth[listProbes].reindex(listOutTumourSamples).values.astype(bool),
            ~dfIsDel[listCNVCols].reindex(listOutTumourSamples).values.astype(bool))

        dfHyperMethNoDel = pd.DataFrame(data=arrayIsHyperMethNoDel,
                                        index=listOutTumourSamples,
                                        columns=listGenes)


        dfClustByTarget = pd.DataFrame(data=np.zeros((4, numTargets), dtype=float),
                                       index=['iClust1', 'iClust2', 'iClust3', 'Total'],
                                       columns=listGenes)
        for strGene in listGenes:
            numHyperMethNoDel = np.sum(dfHyperMethNoDel[strGene])
            dfClustByTarget.loc['Total',strGene] = numHyperMethNoDel/len(listOutTumourSamples)

            for iCluster in range(3):
                listInCluster = dfClusters[dfClusters['Cluster'].str.startswith(f'iClust{iCluster+1}')].index.tolist()
                numHyperMethNoDel = np.sum(dfHyperMethNoDel[strGene].reindex(listInCluster))
                dfClustByTarget.loc[f'iClust{iCluster+1}',strGene] = numHyperMethNoDel/len(listInCluster)

        listBestClusterCombos = []
        for iClust in range(3):
            listInCluster = dfClusters[dfClusters['Cluster'].str.startswith(f'iClust{iClust+1}')].index.tolist()
            setTumourSamples = set(listOutTumourSamples).intersection(set(listInCluster))
            setTargetGenes = set(listGenes)
            # setTargetGenes.remove('CDKN2A')

            listCoveredPatients = []
            listOutGeneOrder = []
            for iGene in range(4):
                listPatientsToCover = list(setTumourSamples.difference(set(listCoveredPatients)))
                listGenesToCover = list(setTargetGenes.difference(set(listOutGeneOrder)))

                dfTargetPatientsForGene = np.sum(dfHyperMethNoDel[listGenesToCover].reindex(listPatientsToCover), axis=0)
                arraySortByPatsCovered = np.argsort(dfTargetPatientsForGene.values.astype(float))

                listGenesByFurtherPatCover = [listGenesToCover[i] for i in arraySortByPatsCovered][::-1]

                strGeneBestCoverage = listGenesByFurtherPatCover[0]
                listOutGeneOrder.append(strGeneBestCoverage)

                listPatientsCoveredByGene = [strPatient for strPatient in listPatientsToCover
                                             if dfHyperMethNoDel.loc[strPatient,strGeneBestCoverage]]

                listCoveredPatients += listPatientsCoveredByGene

            listBestClusterCombos.append(listOutGeneOrder)

        listOfListsGeneCombos = [['BCO2', 'CDKN2A']] + \
                                listBestClusterCombos + \
                                [['HHIP', 'MT1M', 'PZP', 'TTC36']] + \
                                [['CPS1', 'CDKN2A', 'HHIP',
                                  'MT1E', 'MT1M', 'PTGR1',
                                  'TMEM106A']] + \
                                [['BCO2', 'CPS1', 'CDKN2A', 'HHIP',
                                  'MT1E', 'MT1M', 'PSAT1', 'PTGR1',
                                  'PZP', 'TMEM106A', 'TTC36', 'MIMAT0000421']]

        handFig = plt.figure(figsize=(6.5,8))
        handAx = handFig.add_axes([0.15, 0.65, 0.80, 0.30])

        handAx.bar(x=np.arange(numTargets)-0.25,
                   height=dfClustByTarget.loc['iClust1'].values.astype(float),
                   width=0.1,
                   color=Plot.arrayColorMap.to_rgba(0),
                   align='edge')
        handAx.bar(x=np.arange(numTargets)-0.125,
                   height=dfClustByTarget.loc['iClust2'].values.astype(float),
                   width=0.1,
                   color=Plot.arrayColorMap.to_rgba(2),
                   align='edge')
        handAx.bar(x=np.arange(numTargets)+0.025,
                   height=dfClustByTarget.loc['iClust3'].values.astype(float),
                   width=0.1,
                   color=Plot.arrayColorMap.to_rgba(4),
                   align='edge')
        handAx.bar(x=np.arange(numTargets)+0.15,
                   height=dfClustByTarget.loc['Total'].values.astype(float),
                   width=0.1,
                   color='k',
                   align='edge')

        handAx.set_ylim([-0.03, 1])
        handAx.set_yticks([0, 0.5, 1])
        handAx.set_yticklabels(['0%', '50%', '100%'])
        handAx.set_ylabel('Proportion of patients')
        handAx.set_title('Patients with evidence of gene hypermethylation\n and not gene loss/deletion')

        handAx.spines['top'].set_visible(False)
        handAx.spines['right'].set_visible(False)

        handAx.set_xticks(np.arange(len(listOfListsGeneCombos)))
        handAx.set_xticklabels([])
        arrayYLim = handAx.get_ylim()
        numTextYPos = arrayYLim[0] - 0.03*np.ptp(arrayYLim)
        for iGene in range(numTargets):
            if listGenes[iGene].startswith('MIMAT'):
                strGeneToDisp = Plot.dictMicRNAIDToName[listGenes[iGene]]
            else:
                strGeneToDisp = listGenes[iGene]
            handAx.text(iGene, numTextYPos, strGeneToDisp,
                        rotation=90, ha='center', va='top')



        handAxCombo = handFig.add_axes([0.15, 0.15, 0.80, 0.27])

        listSpacer = [-0.25, -0.125, 0.025]
        for iCombo in range(len(listOfListsGeneCombos)):

            arrayIsCoveredByCombo = np.any(dfHyperMethNoDel[listOfListsGeneCombos[iCombo]].values.astype(bool),
                                           axis=1)

            numHyperMethNoDel = np.sum(arrayIsCoveredByCombo)
            handAxCombo.bar(x=iCombo+0.15,
                            height=numHyperMethNoDel/len(listOutTumourSamples),
                            width=0.1,
                            color='k',
                            align='edge')

            for iCluster in range(3):
                listInCluster = dfClusters[dfClusters['Cluster'].str.startswith(f'iClust{iCluster+1}')].index.tolist()
                arrayIsCoveredByCombo = np.any(dfHyperMethNoDel[listOfListsGeneCombos[iCombo]].reindex(listInCluster).values.astype(bool),
                                               axis=1)
                numHyperMethNoDel = np.sum(arrayIsCoveredByCombo)
                handAxCombo.bar(x=iCombo+listSpacer[iCluster],
                                height=numHyperMethNoDel/len(listInCluster),
                                width=0.1,
                                color=Plot.arrayColorMap.to_rgba(iCluster*2),
                                align='edge')

        handAxCombo.set_ylim([-0.03, 1])
        handAxCombo.set_yticks([0, 0.5, 1])
        handAxCombo.set_yticklabels(['0%', '50%', '100%'])
        handAxCombo.set_ylabel('Proportion of patients')
        handAxCombo.set_title('Patients with evidence of gene hypermethylation and not\ngene loss/deletion in at least one target')

        handAxCombo.spines['top'].set_visible(False)
        handAxCombo.spines['right'].set_visible(False)

        handAxCombo.set_xticks(np.arange(len(listOfListsGeneCombos)))
        handAxCombo.set_xticklabels([])
        arrayYLim = handAxCombo.get_ylim()
        numTextYPos = arrayYLim[0] - 0.03*np.ptp(arrayYLim)
        for iCombo in range(len(listOfListsGeneCombos)):
            listInCombo = listOfListsGeneCombos[iCombo].copy()
            for iGene in range(len(listInCombo)):
                if listInCombo[iGene].startswith('MIMAT'):
                    listInCombo[iGene] = Plot.dictMicRNAIDToName[listInCombo[iGene]]

            strOut = ',\n+ '.join(listInCombo[0:-1]) + ',\n or ' + listInCombo[-1]
            # arrayPlusLoc = np.where([charOut == '+' for charOut in strOut])[0]
            # if len(arrayPlusLoc) > 1:
            #     strOut = strOut[0:arrayPlusLoc[1]-1] + '\n' + strOut[arrayPlusLoc[1]:]

            handAxCombo.text(iCombo, numTextYPos, strOut,
                        ha='center', va='top', fontsize=5)

        handFig.text(0.01, 0.98, '(A)', fontweight='bold')
        handFig.text(0.01, 0.47, '(B)', fontweight='bold')


        for strFormat in Plot.listFigFormats:
            handFig.savefig(os.path.join(PathDir.pathPlotFolder, 'SuppFig1.'+strFormat),
                            dpi=300)
        plt.close(handFig)

        return flagResult

    def supp_fig_five(flagResult=False):

        dfInhouseDiffExpr = PreProc.inhouse_dname()

        dfIn = pd.read_csv(os.path.join(PathDir.pathDataFolder, 'HHIP_beta.tsv'), sep='\t',
                           index_col=0, header=0)
        listProbes = dfIn.index.tolist()
        listConds = dfIn.columns.tolist()
        listCondsOrdered = [strCol for strCol in listConds if '_NO_G_B' in strCol] + \
                           [strCol for strCol in listConds if '_A' in strCol] + \
                           [strCol for strCol in listConds if 'NO_G_L' in strCol] + \
                           [strCol for strCol in listConds if '_i' in strCol]

        arrayProbeMean = np.mean(dfIn.values.astype(float), axis=1)

        dfProbeMean = pd.DataFrame(data=np.zeros((len(listProbes),1), dtype=float),
                                   index=listProbes,
                                   columns=['Mean'])
        dfProbeMean.iloc[:,0] = arrayProbeMean

        dfDeltaBeta = pd.DataFrame(data=np.zeros(np.shape(dfIn), dtype=float),
                               index=listProbes,
                               columns=listCondsOrdered)

        for iProbe in range(len(arrayProbeMean)):
            dfDeltaBeta.iloc[iProbe,:] = dfIn.iloc[iProbe,:].values.astype(float) - arrayProbeMean[iProbe]

        numMaxAbsDelBeta = np.max(np.abs(np.ravel(dfDeltaBeta.values.astype(float))))

        handFig = plt.figure(figsize=(6.5, 6.5))
        handAx = handFig.add_axes([0.12, 0.01, 0.63, 0.80])

        handDelBeta = handAx.matshow(dfDeltaBeta.values.astype(float),
                       vmin=-numMaxAbsDelBeta,
                       vmax=numMaxAbsDelBeta,
                       aspect='auto',
                       cmap=plt.cm.PRGn)

        handAx.set_xticks([])
        # handFig.text(x=0.5, y=0.99,
        #              s='HHIP probes',
        #              ha='center', va='top')

        handAx.set_yticks(np.arange(len(listProbes)))
        handAx.set_yticklabels(listProbes, fontsize=Plot.numFontSize*0.75)
        for iCond in range(len(listCondsOrdered)):
            strCond = listCondsOrdered[iCond]
            strCondToDisp = 'Rep ' + strCond[-1]

            handAx.text(iCond, -0.55,
                        strCondToDisp,
                        ha='center', va='bottom',
                        rotation=90)

        numConstructYPos = -15
        handAx.annotate(s='',
                        xy=(0.0, numConstructYPos),
                        xytext=(7.0, numConstructYPos),
                        xycoords='data',
                        annotation_clip=False,
                        arrowprops=dict(linestyle='-',
                                        arrowstyle='-'))
        handAx.text(3.5, numConstructYPos-1, 'SpdCas9-VPR +\nMCP-MS2-p65-HSF1',
                    ha='center', va='bottom',
                    fontsize=Plot.numFontSize*1.5)

        handAx.annotate(s='',
                        xy=(8.0, numConstructYPos),
                        xytext=(15.0, numConstructYPos),
                        xycoords='data',
                        annotation_clip=False,
                        arrowprops=dict(linestyle='-',
                                        arrowstyle='-'))
        handAx.text(11.5, numConstructYPos-1, 'SpdCas9-TET1-CD +\nMCP-MS2-p65-HSF1',
                    ha='center', va='bottom',
                    fontsize=Plot.numFontSize*1.5)


        numGuideYPos = -10
        handAx.annotate(s='',
                        xy=(0.0, numGuideYPos),
                        xytext=(3.0, numGuideYPos),
                        xycoords='data',
                        annotation_clip=False,
                        arrowprops=dict(linestyle='-',
                                        arrowstyle='-'))
        handAx.text(1.5, numGuideYPos-1, 'No gRNA',
                    ha='center', va='bottom',
                    fontsize=Plot.numFontSize*1.5)

        handAx.annotate(s='',
                        xy=(4.0, numGuideYPos),
                        xytext=(7.0, numGuideYPos),
                        xycoords='data',
                        annotation_clip=False,
                        arrowprops=dict(linestyle='-',
                                        arrowstyle='-'))
        handAx.text(5.5, numGuideYPos-1, 'gRNA #4',
                    ha='center', va='bottom',
                    fontsize=Plot.numFontSize*1.5)

        handAx.annotate(s='',
                        xy=(8.0, numGuideYPos),
                        xytext=(11.0, numGuideYPos),
                        xycoords='data',
                        annotation_clip=False,
                        arrowprops=dict(linestyle='-',
                                        arrowstyle='-'))
        handAx.text(9.5, numGuideYPos-1, 'No gRNA',
                    ha='center', va='bottom',
                    fontsize=Plot.numFontSize*1.5)

        handAx.annotate(s='',
                        xy=(12.0, numGuideYPos),
                        xytext=(15.0, numGuideYPos),
                        xycoords='data',
                        annotation_clip=False,
                        arrowprops=dict(linestyle='-',
                                        arrowstyle='-'))
        handAx.text(13.5, numGuideYPos-1, 'gRNA #4',
                    ha='center', va='bottom',
                    fontsize=Plot.numFontSize*1.5)

        arrayVertLine = np.arange(start=3.5, stop=len(listConds), step=4)
        for numXPos in arrayVertLine:
            handAx.axvline(x=numXPos,
                           ymin=0, ymax=1,
                           color='k')

        handAx = handFig.add_axes([0.76, 0.01, 0.05, 0.80])

        handProbeMean = handAx.matshow(dfProbeMean.values.astype(float),
                       vmin=0,
                       vmax=1,
                       aspect='auto',
                       cmap=plt.cm.viridis)

        handAx.set_xticks([])
        handAx.set_yticks([])

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAx.spines[strAxLoc].set_linewidth(0.1)

        listPValCols = [strCol for strCol in dfInhouseDiffExpr.columns.tolist() if strCol.endswith(':adj.P.Val')]

        handAx = handFig.add_axes([0.82, 0.01, 0.06, 0.80])
        handDiffMeth = handAx.matshow(dfInhouseDiffExpr[listPValCols].values.astype(float),
                                       vmin=0,
                                       vmax=1,
                                       aspect='auto',
                                       cmap=plt.cm.plasma_r)

        handAx.set_xticks([])
        handAx.set_yticks([])
        listConstructs=['VPR', 'TET1-CD']
        for iCond in range(len(listConstructs)):
            handAx.text(iCond, -0.7,
                        listConstructs[iCond],
                        rotation=90,
                        ha='center', va='bottom',
                        fontsize=8)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAx.spines[strAxLoc].set_linewidth(0.1)

        handAxCBar = handFig.add_axes([0.91, 0.65, 0.01, 0.1])
        handColorBar = plt.colorbar(handProbeMean, cax=handAxCBar, ticks=[0, 0.5, 1])
        handColorBar.ax.tick_params(labelsize=Plot.numFontSize)
        handAxCBar.set_title('Average\n'+r'${\beta}$', fontsize=Plot.numFontSize)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxCBar.spines[strAxLoc].set_linewidth(0.1)

        strMaxAbsDelBetaClean = '{:03.2f}'.format(numMaxAbsDelBeta)
        numMaxAbsDelBetaClean = float(strMaxAbsDelBetaClean)
        handAxCBar = handFig.add_axes([0.91, 0.45, 0.01, 0.1])
        handColorBar = plt.colorbar(handDelBeta, cax=handAxCBar,
                                    ticks=[-numMaxAbsDelBetaClean, 0, numMaxAbsDelBetaClean],
                                    extend='both')
        handColorBar.ax.tick_params(labelsize=Plot.numFontSize)
        handAxCBar.set_title(r'${\Delta}{\beta}$', fontsize=Plot.numFontSize)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxCBar.spines[strAxLoc].set_linewidth(0.1)


        # strMaxAbsDelBetaClean = '{:03.2f}'.format(numMaxAbsDelBeta)
        # numMaxAbsDelBetaClean = float(strMaxAbsDelBetaClean)
        handAxCBar = handFig.add_axes([0.91, 0.20, 0.01, 0.1])
        handColorBar = plt.colorbar(handDiffMeth, cax=handAxCBar,
                                    ticks=[0, 1],
                                    extend='max')
        handColorBar.ax.tick_params(labelsize=Plot.numFontSize)
        handAxCBar.set_title('adj.\n$p$-val.', fontsize=Plot.numFontSize)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxCBar.spines[strAxLoc].set_linewidth(0.1)


        for strFormat in Plot.listFigFormats:
            handFig.savefig(os.path.join(PathDir.pathPlotFolder, f'SuppFig5.{strFormat}'), dpi=300)

        plt.close(handFig)

        return flagResult

# dfMetaLIHC = TCGAFunctions.lihc_metadata()
# dfRNA = TCGAFunctions.lihc_mess_rna()

# dictGeneSets = TCGAFunctions.int_clust_genesets()
# dfInferredClust = TCGAFunctions.infer_unclassified_clusters(flagPerformInference=True, flagPlotInfClust=True)

# dfMicroRNA = TCGAFunctions.lihc_micro_rna()

# dfDNAme, dictOfDictProbePoint = TCGAFunctions.lihc_dna_me()

# dfCNV = TCGAFunctions.lihc_copy_number()
# dfMut = TCGAFunctions.lihc_mutation()

# _ = PreProc.inhouse_dname()

# dictOrderPatients = PlotFunc.sample_out_order(flagPerformExtraction=True)
#
#
_ = Plot.figure_one()
_ = Plot.supp_fig_one()

_ = Plot.supp_fig_five()