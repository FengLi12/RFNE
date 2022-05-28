
methy<-read.table(file="./methylation_14tumor.txt",header=T,row.names = 1,sep="\t")
mRNA<-read.table(file="./mRNA_14tumor.txt",header=T,row.names = 1,sep="\t")

library(randomForest)
set.seed(7890)
rf <- randomForest(x=methy, importance=TRUE, ntree=500, proximity = TRUE)
SIM_methy_d<- rf$proximity

set.seed(5678)
rf <- randomForest(x=mRNA, importance=TRUE, ntree=500, proximity = TRUE)
SIM_mRNA_d<- rf$proximity


write.table(SIM_methy_d,file = 'HumanMethylation_pair.txt',sep = '\t')
write.table(SIM_mRNA_d,file = 'mRNA_d_pair.txt',sep = '\t')


