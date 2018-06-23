
#Exercicios Aula
set.seed(12)
x <- 32 + 16^2 - 25^3
y <- x/345
uniDist <- runif(30, min=10, max=50)
rm(y)
rm(list = ls())

X <- rpois(n = 100, lambda = 5)

#Exercicio 3
obj = c(54, 0, 17, 94, 12.5, 2, 0.9, 15)
obj + c(5,6)


rept = rep(c("A","B", "C"), times = c(15,12,8) )

caracter <- rept == "B"
caracter

sum(rept == "B")


#3
uniDist <- runif(100, 0,1)
sum(uniDist >= .5)

pot2 <- 2^(1:50)
qdd <- (1:50)^2 

pares <- pot2 == qdd
pares <- ifelse(2^(1:50) == (1:50)^2, qdd, NULL)



sin(seq(0, 2*pi, 0.1))
cos(seq(0, 2*pi, 0.1))
tan(seq(0, 2*pi, 0.1))


tan_pi = sin(seq(0, 2*pi, 0.1))/cos(seq(0, 2*pi, 0.1))
tan_pi
tan(seq(0, 2*pi, 0.1))

tan_pi - tan(seq(0, 2*pi, 0.1))



rpois(n = 100, 5)


fat <- factor(c("alta", "baixa", "baixa", "media", "alta","media", "baixa","media","media"))

unclass(fat)

as.integer(fat)
as.character(fat)


fat <- factor(c("alta", "baixa", "baixa", "media", "alta","media", "baixa","media","media"), 
              levels = c("media", "alta","baixa"))
fat
as.integer(fat)
as.character(fat)




##################################################
############         MATRIZES         ############
##################################################


mat <- matrix(1:12, nrow = 3, ncol = 4)
mat

mat <- matrix(1:12, nrow = 3, ncol = 4, byrow = TRUE)
mat

class(mat)

# Append rows and columns to the column
rbind(mat, rep(99, 4))

# Matrices are essencially vectors
mat <- 1:10
mat
dim(mat) <- c(2,5)
mat

# math operations with matrices should use %*% operator
mat <- matrix(1:12, nrow = 3, ncol = 4, byrow = TRUE)
mat

mat2 <- matrix(1, nrow = 4, ncol = 3)
mat2

mat %*% mat2

#
# DataFrames


df <- data.frame(
          nome = c("Pedro", "Mayra", "Leia"),
          sexo = c("M","F", "F"),
          idade = c(25, 22, 51))
df
class(df)
typeof(df)
dim(df)
str(df)



#
# Exercicios 3
#

mat <- matrix(c(2, 8, 4, 0, 4, 1, 9, 7, 5), ncol = 3, byrow = TRUE)
rownames(mat) <- c("x", "y", "z")
colnames(mat) <- c("a", "b", "c")
mat

objectList <- list(rep = rep(c("A","B", "C"), times = c(2,5,4)), 
                   mat = mat)
objectList

objectList["fator"] <- list(factor(c("brava", "joaquina", "armacao")))
objectList


df <- data.frame( 
      local = c("A","B", "C","D"),
      contagem = c(42, 34, 59, 18))
df

##



user <- data.frame( 
              nome = "Pedro",
              sobrenome = "Neto",
              animalEstimacao = "Sim",
              qtdAnimalEstimacao = 4)
user

user <- rbind(user, data.frame(nome = "Mayra", sobrenome = "Levandoski", animalEstimacao = "Nao", qtdAnimalEstimacao = 0))
user <- rbind(user, data.frame(nome = "Joao Gabriel", sobrenome = "Gomes", animalEstimacao = "sim", qtdAnimalEstimacao = 1))
user <- rbind(user, data.frame(nome = "Wiliam", sobrenome = "Martins", animalEstimacao = "sim", qtdAnimalEstimacao = 1))

user



valores = c(88, 5, 12, 13)
valores[3]
valores[ valores i 88]










