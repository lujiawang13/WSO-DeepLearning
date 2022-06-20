import numpy as np

def predict_f(model_info,TestData):
    model=model_info['model']
    bs=model_info['bs']
    K = model_info['K']
    features=model_info['features']

    xTest=TestData.loc[:,features]
    predictions = model.predict(xTest)
    labels=labels_f(predictions, K, bs)
    return labels

def labels_f(predictions,K,bs):
    npred=len(predictions)
    predictions=predictions.reshape(npred,)
    labels=np.repeat(np.nan, npred, axis=0)
    #Class K-1
    for k in range(K-1):
        if k==0:
            indx = np.where(predictions > bs[k])
        else:
            indx = np.where((predictions > bs[k]) & (predictions < bs[k - 1]))
        labels[indx]=k+1
    # class K
    indx = np.where(predictions < bs[K - 2])
    labels[indx] = K
    return labels
