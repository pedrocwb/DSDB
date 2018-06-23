
#par(mfrow=c(1,2))

data <- read.csv("youtube.txt", sep = " ", header = TRUE)
head(data)


inventonahora <- data[data$CANAL == "inventonahora", ]
vocesabia     <- data[data$CANAL == "vocesabia", ]

factor <- 100000

inventonahora$INSCR_CUMSUM  <- cumsum(inventonahora$INSCRITOS/factor)
vocesabia$INSCR_CUMSUM      <- cumsum(vocesabia$INSCRITOS/factor)

# Modelo Logistico de crescimento de bactérias
f_logistic <- function(par, dias, insc){
  mu <- par[1] / (1 + exp(par[2] * ( dias - par[3])))
  SQ_logit <- sum( (insc - mu)^2 )
  return(SQ_logit)
}

##### CANAL INVENTO NA HORA #####

# Minimização da função
logisticMinimization <- optim(par = c(round(max(inventonahora$INSCR_CUMSUM)), 0, mean(inventonahora$INSCR_CUMSUM)), 
             fn = f_logistic, dias = inventonahora$DIAS, insc = inventonahora$INSCR_CUMSUM)
logisticMinimization

# Previsão do numero acumulado de inscritos no canal nos próximos 365 dias  
dias <- seq(min(inventonahora$DIAS), max(inventonahora$DIAS) + 365, by=1)
preds <- logisticMinimization$par[1] / (1 + exp(logisticMinimization$par[2] * ( dias - logisticMinimization$par[3])))

# Plota gráfico de previsao
plot(dias, preditos,col="Red", type = "l")
lines(inventonahora$DIAS, inventonahora$INSCR_CUMSUM)
abline(v=max(inventonahora$DIAS))




##### CANAL VOCE SABIA? #####

# Otimizaçao da funçao
ols <- optim(par = c(round(max(vocesabia$INSCR_CUMSUM)), 0, mean(vocesabia$INSCR_CUMSUM)), 
             fn = f_logistic, dias = vocesabia$DIAS, insc = vocesabia$INSCR_CUMSUM)
ols

# Previsão do numero acumulado de inscritos no canal nos próximos 365 dias  
dias <- seq(min(vocesabia$DIAS), max(vocesabia$DIAS) + 365, by=1)
preditos <- ols$par[1] / (1 + exp(ols$par[2] * ( dias - ols$par[3])))

plot(dias, preditos,col="Red", type = "l")
lines(vocesabia$DIAS, vocesabia$INSCR_CUMSUM)
abline(v=max(vocesabia$DIAS))








