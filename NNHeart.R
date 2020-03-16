normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


heart <- read.csv("~/HeartDisease/heart.csv")

dfNorm <- as.data.frame(lapply(heart["age"], normalize))
heart["age"] <- dfNorm
heart["age"] <- as.data.frame(lapply(heart["age"], function(x){replace(x, between(x, 0.0, 0.25), 0.1)}))
heart["age"] <- as.data.frame(lapply(heart["age"], function(x){replace(x, between(x, 0.25, 0.6), 0.4)}))
heart["age"] <- as.data.frame(lapply(heart["age"], function(x){replace(x, between(x, 0.5, 0.75), 0.6)}))
heart["age"] <- as.data.frame(lapply(heart["age"], function(x){replace(x, between(x, 0.75, 1), 0.9)}))

dfNorm <- as.data.frame(lapply(heart["trestbps"], normalize))
heart["trestbps"] <- dfNorm
heart["trestbps"] <- as.data.frame(lapply(heart["trestbps"], function(x){replace(x, between(x, 0.0, 0.33), 0.2)}))
heart["trestbps"] <- as.data.frame(lapply(heart["trestbps"], function(x){replace(x, between(x, 0.33, 0.67), 0.6)}))
heart["trestbps"] <- as.data.frame(lapply(heart["trestbps"], function(x){replace(x, between(x, 0.67, 1), 1)}))

dfNorm <- as.data.frame(lapply(heart["chol"], normalize))
heart["chol"] <- dfNorm
heart["chol"] <- as.data.frame(lapply(heart["chol"], function(x){replace(x, between(x, 0.0, 0.2), 0.1)}))
heart["chol"] <- as.data.frame(lapply(heart["chol"], function(x){replace(x, between(x, 0.2, 0.4), 0.3)}))
heart["chol"] <- as.data.frame(lapply(heart["chol"], function(x){replace(x, between(x, 0.4, 0.6), 0.5)}))
heart["chol"] <- as.data.frame(lapply(heart["chol"], function(x){replace(x, between(x, 0.6, 0.8), 0.7)}))
heart["chol"] <- as.data.frame(lapply(heart["chol"], function(x){replace(x, between(x, 0.8, 1), 0.9)}))

dfNorm <- as.data.frame(lapply(heart["thalach"], normalize))
heart["thalach"] <- dfNorm

dfNorm <- as.data.frame(lapply(heart["cp"], normalize))
heart["cp"] <- dfNorm
heart["cp"] <- as.data.frame(lapply(heart["cp"], function(x){replace(x, between(x, 0.0, 0.25), 0.9)}))
heart["cp"] <- as.data.frame(lapply(heart["cp"], function(x){replace(x, between(x, 0.25, 0.5), 0.9)}))
heart["cp"] <- as.data.frame(lapply(heart["cp"], function(x){replace(x, between(x, 0.5, 0.75), 0.1)}))
heart["cp"] <- as.data.frame(lapply(heart["cp"], function(x){replace(x, between(x, 0.75, 1), 0.01)}))


dfNorm <- as.data.frame(lapply(heart["thal"], normalize))
heart["thal"] <- dfNorm
heart["thal"] <- as.data.frame(lapply(heart["thal"], function(x){replace(x, between(x, 0.0, 0.33), 0.2)}))
heart["thal"] <- as.data.frame(lapply(heart["thal"], function(x){replace(x, between(x, 0.33, 0.67), 0.6)}))
heart["thal"] <- as.data.frame(lapply(heart["thal"], function(x){replace(x, between(x, 0.67, 1), 0.9)}))

dfNorm <- as.data.frame(lapply(heart["slope"], normalize))
heart["slope"] <- dfNorm

dfNorm <- as.data.frame(lapply(heart["ca"], normalize))
heart["ca"] <- dfNorm

heart["sex"] <- as.data.frame(lapply(heart["sex"], function(x){replace(x, x == 0, 0.1)}))
heart["sex"] <- as.data.frame(lapply(heart["sex"], function(x){replace(x, x == 1, 0.9)}))

heart["fbs"] <- as.data.frame(lapply(heart["fbs"], function(x){replace(x, x == 0, 0.1)}))
heart["fbs"] <- as.data.frame(lapply(heart["fbs"], function(x){replace(x, x == 1, 0.9)}))

heart["restecg"] <- as.data.frame(lapply(heart["restecg"], function(x){replace(x, x == 0, 0.1)}))
heart["restecg"] <- as.data.frame(lapply(heart["restecg"], function(x){replace(x, x == 1, 0.9)}))

heart["exang"] <- as.data.frame(lapply(heart["exang"], function(x){replace(x, x == 0, 0.1)}))
heart["exang"] <- as.data.frame(lapply(heart["exang"], function(x){replace(x, x == 1, 0.9)}))

smp_size <- floor(0.75 * nrow(heart))
train_ind_rand <- sample(seq_len(nrow(heart)), size = smp_size)

trainrand <- heart[train_ind_rand, ]
testrand <- heart[-train_ind_rand, ]

trainseq <- heart[1:227, ]
testseq <- heart[227:303, ]

write.csv(heart, "~/HeartDisease/heart1.csv", row.names = FALSE)
write.csv(trainrand, "~/HeartDisease/trainrand.csv", row.names = FALSE)
write.csv(testrand, "~/HeartDisease/testrand.csv", row.names = FALSE)
write.csv(trainseq, "~/HeartDisease/trainseq.csv", row.names = FALSE)
write.csv(testseq, "~/HeartDisease/testseq.csv", row.names = FALSE)
