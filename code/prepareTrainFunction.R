prepareTrainFunction = function(noOfTuningParams) {
  set.seed(1324)
  seeds <- vector(mode = "list", length = 11)
  for(i in 1:10) {
    seeds[[i]] = sample.int(n=1000, noOfTuningParams)
  }
  seeds[[11]] = sample.int(n=1000, 1)
  
  cv.trainControl = trainControl(method = "adaptive_cv",
                                 number = 10,
                                 verboseIter = TRUE,
                                 index = createFolds(model.pca.projected$classe),
                                 seeds = seeds,
                                 savePredictions = TRUE,
                                 allowParallel = TRUE)
}