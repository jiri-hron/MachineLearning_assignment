cleanData = function(path) {
  if(!require(caret)) {
    stop("install caret and dependent packages before running")
  }
  
  ## read in the file
  data = read.csv(path,
                      header = TRUE, sep = ",",
                      comment.char = "",
                      na.strings = "NA",
                      stringsAsFactors = FALSE)
  
  ## throw away all observations marked as new window, which contain only
  ## aggregate statistics on the whole preceding time window of 2.5 secs
  ## these are not available in testing set
  data = data[data$new_window == "no",]
  
  ## throw away variables with inappropriate and useless variables
  data = data[, !(colnames(data) %in%
                            c("X", "user_name", 
                              "raw_timestamp_part_1",
                              "raw_timestamp_part_2", 
                              "cvtd_timestamp",
                              "new_window", "num_window"))]
  
  ## get rid of all NA columns and all empty char columns right away
  noOfNA = apply(
    apply(X = data, MARGIN = 2, is.na),
    MARGIN = 2,
    sum)
  noOfEmptyChars = apply(
    apply(X = data, MARGIN = 2, function(x) x == ""),
    MARGIN = 2,
    sum)
  keep = noOfNA < nrow(data) & noOfEmptyChars < nrow(data)
  data = data[,keep]
  
  ## only the response variable is categorical
  data[,"classe"] = as.factor(data[,"classe"])
  
  ## return preprocessed data
  data
}