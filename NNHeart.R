normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


heart <- read.csv("~/Downloads/HeartDisease/heart.csv")

max(heart["chol"])
min(heart["chol"])

dfNorm <- as.data.frame(lapply(heart["age"], normalize))
heart["age"] <- dfNorm

dfNorm <- as.data.frame(lapply(heart["trestbps"], normalize))
heart["trestbps"] <- dfNorm

#To be changed
dfNorm <- as.data.frame(lapply(heart["chol"], normalize))
heart["chol"] <- dfNorm

dfNorm <- as.data.frame(lapply(heart["thalach"], normalize))
heart["thalach"] <- dfNorm

#To be changed
dfNorm <- as.data.frame(lapply(heart["cp"], normalize))
heart["cp"] <- dfNorm

#To be changed
dfNorm <- as.data.frame(lapply(heart["thal"], normalize))
heart["thal"] <- dfNorm

dfNorm <- as.data.frame(lapply(heart["slope"], normalize))
heart["slope"] <- dfNorm

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

write.csv(heart, "~/Downloads/HeartDisease/heart1.csv", row.names = FALSE)
write.csv(trainrand, "~/Downloads/HeartDisease/trainrand.csv", row.names = FALSE)
write.csv(testrand, "~/Downloads/HeartDisease/testrand.csv", row.names = FALSE)
write.csv(trainseq, "~/Downloads/HeartDisease/trainseq.csv", row.names = FALSE)
write.csv(testseq, "~/Downloads/HeartDisease/testseq.csv", row.names = FALSE)

max(heart["chol"])
min(heart["chol"])
