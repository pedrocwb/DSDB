# Aluno: Pedro Martins Moreira Neto
# Matricula: 201800077493
# pedromartins.cwb@gmail.com


## Insira aqui o n'mero da sua matrícula para fixar uma semente
## 201800077493
matricula <- "77493"
## Gera 1 milh?o de n?meros aleat?rios de uma distribui??o normal
set.seed(matricula)
pop <- rnorm(n = 1e6, mean = 100, sd = sqrt(200))
## Retira uma amostra aleat?ria de tamanho n = 1000 da popula??o
amostra <- pop[sample(1:length(pop), size = 1000)]


## Fun??es auxiliares
abs.diff <- function(estimationMean, populationMean) {
  return(abs(estimationMean - populationMean))
}

display.histogram <- function(est, m){
  meanpop <- mean(pop)
   hist(est, col = "red", 
        xlab = "Estimations",
        main = c("Histograma de Estimativas m = ", m))
  # insere a linha de media no histograma
   abline(v = meanpop,           
         col = "royalblue",
         lwd = 2)
}

### Exerc?cio 1
# Algoritmo geral para desenvolver o m?todo, considerando o tamanho da amostra (m) fixo:
# 1. Com os dados da amostra, gere uma nova amostra aleat?ria (com reposi??o) de tamanho m = 500.
# 2. Calcule a m?dia dessa nova amostra.
# 3. Repita esse procedimento r = 100000 vezes.
# 4. Fa?a um histograma das r estimativas, calcule a m?dia e compare com a m?dia verdadeira.

bootstrap.fixed.sample <- function (m = 500, r = 100000){
  #configuracao para quantidade de histogramas a serem plotados
  
  par(mfrow = c(1,1))
  # Aloca todo o vetor de uma vez na memoria para que 
  # os dados estejam alinhados e a performance do algoritmo melhore.
  estimations <- numeric(r)
  
  for( i in seq(1:r)){
    sample <- amostra[sample(1:length(amostra), size = m, replace=TRUE)]
    estimations[i] <- c(mean(sample))
  }
  
  display.histogram(estimations, m)
  cat("\n m = ",m ,  " Diferenca Absoluta: ", abs.diff(mean(estimations), mean(pop)))
}

bootstrap.fixed.sample()

# Exerc?cio 2
# Algoritmo geral para desenvolver o m?todo, considerando o tamanho da amostra (m) variando entre
# quatro valores diferentes:
# 1. Com os dados da amostra, gere uma nova amostra aleat?ria (com reposi??o) com tamanhos:
#   m = 100, 300, 500, 700.
# 2. Calcule a m?dia dessa nova amostra, para cada valor de m.
# 3. Repita esse procedimento r = 100000 vezes, para cada valor de m.
# 4. Fa?a um histograma das r estimativas, calcule a m?dia e compare com a m?dia verdadeira (para
#                                                                                            cada valor de m).

bootstrap.variable.sample <- function(m = c(100, 300, 500, 700), r = 100000) {
  #configuracao para quantidade de histogramas a serem plotados
  par(mfrow = c(2,2))
  
  for(each in m){
    estimations <- numeric(r)
    
    for( i in seq(1:r)){
      sample <- amostra[sample(1:length(amostra), size = m, replace=TRUE)]
      estimations[i] <- c(mean(sample))
    }
    
    display.histogram(estimations, each)
    cat("\n m = ",each,  " Diferenca Absoluta: ", abs.diff(mean(estimations), mean(pop)))
  }
}

bootstrap.variable.sample()


