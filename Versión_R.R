# Cargamos los paquetes que vamos a usar
library("dplyr")
library("data.table")
library("caret")
library("pROC")
library("ggplot2")
library("fastDummies")

#Cargamos el dataset
datos <- fread("./trabajo_practico/personality_dataset.csv", sep = ";")

#Exploramos el dataset
str(datos)
summary(datos)
colSums(is.na(datos))
sum(is.na(datos))

#Verificar si los sujetos que tienen nulos son los mismos o diferentes
filas_con_nulos <- datos %>% filter(!complete.cases(.))
View(filas_con_nulos)


#Rellenamos los datos faltantes con la mediana, ya que es una medida más robusta
datos$Time_spent_Alone[is.na(datos$Time_spent_Alone)] <- median(datos$Time_spent_Alone, na.rm = TRUE)
datos$Going_outside [is.na(datos$Going_outside )] <- median(datos$Going_outside , na.rm = TRUE)
datos$Post_frequency [is.na(datos$Post_frequency )] <- median(datos$Post_frequency , na.rm = TRUE)
datos$Social_event_attendance [is.na(datos$Social_event_attendance )] <- median(datos$Social_event_attendance , na.rm = TRUE)
datos$Friends_circle_size  [is.na(datos$Friends_circle_size  )] <- median(datos$Friends_circle_size  , na.rm = TRUE)

#Verificamos que se hayan eliminado correctamente 
colSums(is.na(datos))
sum(is.na(datos))

#Analizamos la disperción de los datos, obsevamos que no presentan outliers 
boxplot(datos$Time_spent_Alone)
boxplot(datos$Social_event_attendance)
boxplot(datos$Going_outside)
boxplot(datos$Friends_circle_size)
boxplot(datos$Post_frequency)


#Analizamos las variables cualitativas
barplot(table(as.factor(datos$Stage_fear)))
barplot(table(as.factor(datos$Drained_after_socializing)))
barplot(table(as.factor(datos$Personality)))

#Eliminar los vacios en variables cualitativas y luego pasar a factor
datos[datos == ""] <- NA
sum(is.na(datos))
datos <- na.omit(datos)

#Comprobamos que todos los valores hayan sido eliminados correctamente
sum(is.na(datos))
barplot(table(as.factor(datos$Stage_fear)))
barplot(table(as.factor(datos$Drained_after_socializing)))

#Pasamos las variables cualitativas tipo char a factor y 
#pasamos los valores de la ariable objetivo a 1 y 0 para evitar posibles problemas
#a la hora de realizar la matriz de confusión 
datos$Stage_fear <- factor(datos$Stage_fear)
datos$Drained_after_socializing <- factor(datos$Drained_after_socializing)
datos$Personality <- factor(datos$Personality)
datos$Personality <- factor(ifelse(datos$Personality == "Extrovert", 1, 0),
                            levels = c(0, 1))

#Varificamos que todos los datos se hallan modificado correctamente
str(datos)
summary(datos)

#write.csv(datos, file = "Personalidad.csv", row.names = FALSE)
#--------------------------OBJETIVO--------------------------------------
#Tenemos un dataset que muestra la personalidad de distintitos individuos, introvertidos
# o extrovertidos y diferentes habitos socialies y/o personales de los mismos.

#¿Cuales de estás variables están más relacionadas con la personalidad del individuo?

#Utilizaremos regresión logistica para clasificar la personalidad de los individuos
#según sus habitos.

#Aplicamos Dummies sobre las variables tipo factor menos la objetivo
datos <- dummy_cols(datos, 
                   select_columns = c("Drained_after_socializing","Stage_fear"),
                   remove_selected_columns = T)


#Inicializa la semilla aleatoria
set.seed(1234)

#Primera validacion: 70% para el entrenamiento
porc_1 <- 0.7

# Encontramos el n?mero de observaciones de los datos
N <- nrow(datos)

#Encontramos el tama?o de la partici?n de entrenamiento
tamanio <- floor(porc_1*N)

# Encontramos el conjunto de indices de entrenamiento
train.ind <- sample(seq_len(N), size = tamanio)

# Separamos los datos
datos.train <- datos[train.ind,]
datos.test <- datos[-train.ind,]

# Vemos la proporción de la clase
table(datos.train$Personality)
table(datos.test$Personality)

# Encontramos el modelo de reg logística
modelo.datos <- glm(Personality ~ ., 
                    datos.train, 
                    family = "binomial")
summary(modelo.datos)

#Se observan nulos en Drained_after_socializing
#Comprobamos que Drained_after_socializing este correcta y no tenga nulos
summary(datos.train$Drained_after_socializing_No)
summary(datos.train$Stage_fear_No)
summary(datos.train$Stage_fear_Yes)
#Esto quiere decir que las variables fueron excluida automáticamente del modelo por colinealidad perfecta
#Es decir, hay una relación lineal exacta entre esa variable y otra(s) del modelo,
#por lo que no aporta información nueva.


# Graficamos para visualizar una posible relacion entre variables
ggplot(datos.train, aes(x = Time_spent_Alone, y = Friends_circle_size, color = Personality)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(x = "Tiempo solo", y = "Tamaño del círculo de amigos", title = "Dispersión por personalidad")


# Vemos la proporción de la clase
table(datos.train$Personality)
table(datos.test$Personality)

# Predecimos
datos.pred <- predict(modelo.datos, datos.test, type = "response")

# Como nos devuelve la probabilidad, le decimos que si es mayor que 0.5 es 1 y
# si no, es 0
pred.modelo <- ifelse(datos.pred > 0.5,1,0)

# confusionMatrix de caret

datos.conf <- confusionMatrix(factor(pred.modelo),
                                      factor(datos.test$Personality))

datos.conf$table
table(datos.test$Personality)
# 
datos.conf$byClass
datos.conf

# Graficamos la curva ROC y obtenemos el área bajo la curva
roc_obj <- roc(datos.test$Personality, datos.pred, plot = TRUE, col = "blue")
auc(roc_obj)

#Vamos a volver a aplicar el modelo con las variables que son significativas
# Encontramos el modelo de reg logística
modelo.datos_opt <- glm(Personality ~ Time_spent_Alone + Drained_after_socializing_No +Social_event_attendance 
                        + Going_outside + Friends_circle_size, 
                    datos.train, 
                    family = "binomial")
summary(modelo.datos_opt)


#Predecimos
datos.pred_opt <- predict(modelo.datos_opt, datos.test, type = "response")

# Como nos devuelve la probabilidad, le decimos que si es mayor que 0.5 es 1 y
# si no, es 0
modelo.datos_opt <- ifelse(datos.pred_opt > 0.5,1,0)

# confusionMatrix de caret

datos.conf_opt <- confusionMatrix(factor(modelo.datos_opt),
                              factor(datos.test$Personality))

datos.conf_opt$table
table(datos.test$Personality)
# 
datos.conf_opt$byClass
datos.conf_opt

# Graficamos la curva ROC y obtenemos el área bajo la curva
roc_obj <- roc(datos.test$Personality, datos.pred_opt, plot = TRUE, col = "blue")
auc(roc_obj)
