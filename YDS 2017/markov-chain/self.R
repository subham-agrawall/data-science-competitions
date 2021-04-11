# Load data and libraries -------------------------------------------------

library(data.table)
library(markovchain)
library(clickstream)

train <- fread("train.csv")
test <- fread("test.csv")

head(train)
head(test)

train <- train[order(PID)]
test <- test[order(PID)]

# Create list of events per PID such that event sequence is mainta --------

list_train <- train[,.(list(Event)),.(PID,Date)]
list_one <- list_train[,.(list(V1)),.(PID)]
list_one[,V1 := lapply(V1, unlist, use.names = F)]
setnames(list_one,"V1","Events")

prediction <- list()

for(x in 2:2)
{
  PID <- list_one[x,PID]
  events_x <- as.character(unlist(list_one[x,Events]))
  alp= PID
  if (length(events_x)>100){
    events_x=tail(events_x,100)
  }
  alp = paste(alp,toString(events_x), sep = ",")
  alp=c(alp,temp)
  csf<-tempfile()
  writeLines(alp, csf)
  cls <- readClickstreams(csf, header = TRUE)
  mc <- fitMarkovChain(cls,order=2)
  startPattern <- new("Pattern", sequence = events_x)
  a=predict(mc, startPattern, dist=10)
  prediction[[PID]] <- a@sequence
}
