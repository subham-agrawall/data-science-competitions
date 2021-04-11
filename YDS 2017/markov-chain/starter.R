# This is a first order markov chain model. This script should give you fair idea about how to get started with MC models. 
# You can try building second, third order mc models. 


# Load data and libraries -------------------------------------------------

library(data.table)
library(markovchain)

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


# Building Markov Chain Model on PID Level --------------------------------

prediction <- list()

for(x in 1:nrow(list_one))
{
  PID <- list_one[x,PID]
  events_x <- as.character(unlist(list_one[x,Events]))
  
  mcX <- markovchainFit(events_x, method = "mle")
  pred <- predict(object = mcX$estimate, newdata = events_x, n.ahead=10) # predict next 10 events
  
  prediction[[PID]] <- pred
  
}

# Creating final submission file

final_prediction <- data.table(PID = names(prediction), Event = prediction)

for(i in 1:nrow(final_prediction))
{
  for(j in 1:10)
  {
    final_prediction[[paste0("Event",j)]] <- lapply(final_prediction$Event,'[',j)
  }
}

final_prediction[,Event := NULL]
fwrite(final_prediction,"markov_preds.csv")