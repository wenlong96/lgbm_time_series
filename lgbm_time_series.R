# Data preprocessing
library(fpp2)
library(lubridate)
library(dplyr)

df <- read.csv("C://Users//wl//Desktop//Work//[Website]//Demand Forecast//Kaggle Challenge//train.csv")

View(df) # 2013-01-01 to 2017-12-31

unique(df$store) # 10 stores
unique(df$item) # 50 items

df$year <- as.factor(year(df$date)) # add in year column
df$month <- as.factor(month(df$date)) # add in month column
df$day <- as.factor(day(df$date)) # add in day column
df$item <- as.factor(df$item) # change item type to factors
df$store <- as.factor(df$store) # change store type to factors

df[!complete.cases(df), ] # find how many NA rows

x_var_time <- c("year", "month", "day") # assign time variables
x_var <- c("store", "item") # assign other non-time variables
y_var <- c("sales") # assign target variables

df2 <- df |>
  select(all_of(c(x_var, x_var_time, y_var))) |> # filter dataframe
  arrange(year, month, day)


# Create time series object for each item (to extract seasonalities)
library(parallel)
library(doParallel)

df_mean <- df2 |>
  group_by(year, month, day, item) |>
  summarise(mean_sales = mean(sales)) |>
  inner_join(df2, by = c("year", "month", "day", "item")) |>
  filter(store == "1") |>
  select(-c(store, sales))

ncores <- detectCores() - 1
my_cluster <- makeCluster(ncores, type = "PSOCK")
registerDoParallel(cl = my_cluster)


# creating the TS using parallelization and saving all TS objects into a list
TS_list <- foreach(i = 1:length(unique(df$item)), .packages = "dplyr") %dopar% {
  df_sub <- subset(df_mean, item == unique(df$item)[i])
  
  ts(df_sub |>
       arrange(year, month, day) |>
       ungroup() |>
       select(mean_sales), frequency = 365.25)  
}


# extract each of the TS object in the TS_list into its own TS object
for (i in 1:length(unique(df$item))) {
  df_name <- paste("TS_item_", i, sep = "")
  
  assign(df_name, TS_list[[i]])
}


# Exploratory data analysis
library(ggplot2)
library(plotly)
library(scales)
library(lubridate)
library(ggthemes)


# plot for item 1
start_date <- as.Date("2013-01-01")
end_date <- as.Date("2017-12-31")
test1 <- data.frame(Date = 1:length(TS_item_1), Value = TS_item_1)
test1$Date <- start_date + test1$Date - 1

test_plot1 <- ggplot(data = test1, aes(x = Date, y = mean_sales)) +
  geom_line(color = "#009E73") +
  scale_x_date(labels = date_format("%Y-%b"), breaks = seq(from = start_date, to = end_date, by = "4 month")) +
  xlab("Date") +
  ylab("Average Revenue ($)") +
  ggtitle("Average Revenue Per Day for Item 1")

test_plot1


# plot for item 2
start_date <- as.Date("2013-01-01")
end_date <- as.Date("2017-12-31")
test2 <- data.frame(Date = 1:length(TS_item_2), Value = TS_item_2)
test2$Date <- start_date + test2$Date - 1

test_plot2 <- ggplot(data = test2, aes(x = Date, y = mean_sales)) +
  geom_line(color = "#CC79A7") +
  scale_x_date(labels = date_format("%Y-%b"), breaks = seq(from = start_date, to = end_date, by = "4 month")) +
  xlab("Date") +
  ylab("Average Revenue ($)") +
  ggtitle("Average Revenue Per Day for Item 2")

test_plot2


# plot for item 3
start_date <- as.Date("2013-01-01")
end_date <- as.Date("2017-12-31")
test3 <- data.frame(Date = 1:length(TS_item_3), Value = TS_item_3)
test3$Date <- start_date + test3$Date - 1

test_plot3 <- ggplot(data = test3, aes(x = Date, y = mean_sales)) +
  geom_line(color = "#87CEEB") +
  scale_x_date(labels = date_format("%Y-%b"), breaks = seq(from = start_date, to = end_date, by = "4 month")) +
  xlab("Date") +
  ylab("Average Revenue ($)") +
  ggtitle("Average Revenue Per Day for Item 2")

test_plot3


# overlay them
test_plot_all <- ggplot() +
  geom_line(data = test1, aes(x = Date, y = mean_sales, color = "item 1")) +
  geom_line(data = test2, aes(x = Date, y = mean_sales, color = "item 2")) +
  geom_line(data = test3, aes(x = Date, y = mean_sales, color = "item 3")) +
  scale_x_date(labels = date_format("%Y-%b"), breaks = seq(from = start_date, to = end_date, by = "5 month")) +
  xlab("Date") +
  ylab("Average Revenue ($)") +
  ggtitle("Average Revenue Per Day for Item 1, 2, and 3") +
  scale_color_manual(values = c("item 1" = "#009E73", "item 2" = "#CC79A7", "item 3" = "#87CEEB"), name = "Item Type") +
  theme_economist()

ggplotly(test_plot_all, viewer = "new")


# Time series decomposition and seasonality extraction
library(imputeTS)


# multiplicative decomposition
plot(decompose(TS_item_1, type = "multiplicative"))
plot(decompose(TS_item_3, type = "multiplicative"))


# extract normalized seasonal components into a list
seasonal_list <- foreach(i = 1:length(TS_list), .combine = "c", .packages = "imputeTS") %dopar% {
  na_replace(as.numeric(decompose(TS_list[[i]], type = "multiplicative")$seasonal), summary(as.numeric(decompose(TS_list[[i]], type = "multiplicative")$seasonal))[[3]]) / max(na.omit(as.numeric(decompose(TS_list[[i]], type = "multiplicative")$seasonal)))
}

stopCluster(cl = my_cluster)


# add seasonal component into df_mean and rename the column
df_mean_w_seas <- cbind(df_mean, seasonal_list)
names(df_mean_w_seas)[length(names(df_mean_w_seas))] <- "seasonal" 


# add back seasonal component to original dataframe df
df_w_seas <- df |>
  left_join(df_mean_w_seas, by = c("year", "month", "day", "item"))


# Feature engineering
library(tidyr)


# select initial variables for modelling
df_w_seas_filtered <- df_w_seas |> select(c(date, sales, seasonal, store, item, month, day))


# plotting PACF
pacf(df_w_seas_filtered$sales, lag = 60) # we see that day-of-the-week effect and the previous-day effect are the most important


# feature engineering
df_w_seas_filtered <- df_w_seas_filtered |>
  arrange(store, item, date) |>
  group_by(store, item) |>
  mutate(running_days = row_number()) |>
  mutate(lag1d = lag(sales, default = 0)) |>
  mutate(lag7d = lag(sales, n = 7, default = 0)) |>
  fill(lag1d, .direction = "up") |>
  fill(lag7d, .direction = "up") |>
  mutate(day_of_month = case_when(
    day(date) == 1 ~ "start",
    day(date) == days_in_month(date) ~ "end",
    TRUE ~ "normal"
    )) |>
  ungroup() |>
  cbind(data.frame(weekday = factor(wday(df_w_seas_filtered$date, label=TRUE), ordered = FALSE))) |>
  mutate(is_weekend = case_when(
    weekday %in% c("Sat", "Sun") ~ "yes",
    TRUE ~ "no"
  )) |>
  mutate_if(is.character, as.factor) |>
  select(c(date, sales, lag1d, lag7d, seasonal, store, item, month, day, day_of_month, weekday, is_weekend, running_days))

df_w_seas_filtered$date <- as.Date(df_w_seas_filtered$date)


# final check before tuning and modelling
View(df_w_seas_filtered)


# Create training and test set for modelling
library(purrr)


# create a training set with 80% of data
train_df <- df_w_seas_filtered |>
  group_by(store, item) |>
  filter(date >= as.Date("2013-01-01") & date <= as.Date("2016-12-31")) |>
  ungroup()


# filter out value for last day of training set for lag1d of first day of test set
train_sales_2016 <- train_df |>
  filter(date == as.Date("2016-12-31")) |>
  select(store, item, sales)

train_sales_7days_before <- train_df |>
  filter(date >= as.Date("2016-12-25") & date <= as.Date("2016-12-31")) |>
  select(date, store, item, sales) |>
  mutate(date = date + days(7))
  

# create the test set with the remaining 20% of data
test_df <- df_w_seas_filtered |>
  group_by(store, item) |>
  filter(date > as.Date("2016-12-31")) |>
  ungroup() |>
  left_join(train_sales_2016, by = c("store", "item"), suffix = c("", ".y")) |>
  mutate(lag1d = ifelse(date == as.Date("2017-01-01"), sales.y, 0.0001),
         lag1d = replace(lag1d, is.na(lag1d), sales)) |>
  select(-sales.y) |>
  left_join(train_sales_7days_before, by = c("store", "item", "date"), suffix = c("", ".y")) |>
  mutate(lag7d = ifelse(date >= as.Date("2017-01-01") & date <= as.Date("2017-01-07"), sales.y, 0.0001),
         lag7d = replace(lag7d, is.na(lag7d), sales)) |>
  select(-sales.y, -date)


# duplicate test set without target variable
test_df2 <- test_df[, colnames(test_df) != "sales"]


# extract target variable of test set
y_test <- as.data.frame(test_df) |>
  select(sales) |>
  pull(sales) |>
  as.numeric()


# remove date and the first 7 days of each store and item combination as there is no observable lag7d values
train_df <- train_df |>
  filter(date >= as.Date("2013-01-08")) |>
  select(-date)


# duplicate training set without target variable
train_df2 <- train_df[, colnames(train_df) != "sales"]


# extract target variable of training set
y_train <- as.data.frame(train_df) |>
  select(sales) |>
  pull(sales) |>
  as.numeric()


# Hyperparameter tuning
set.seed(333)
library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(paradox)
library(mlr3tuning)
library(mlr3mbo)
library(DiceKriging)
library(ranger)


# create task for lgbm
target_col <- "sales"

task <- mlr3::TaskRegr$new(
  id = "demand_forecast",
  backend = train_df,
  target = target_col
)


# editing the learner and set hyperparameter search bounds
mlr_learners$get("regr.lightgbm")
learner <- mlr3extralearners::lrn(
  "regr.lightgbm",
  bagging_fraction = to_tune(0.4, 1),
  num_leaves = to_tune(7, 4095),
  max_depth = to_tune(2, 63),
  learning_rate = to_tune(0.001, 0.25),
  drop_rate = to_tune(0.01, 0.2),
  skip_drop = to_tune(0, 0.5)
)


# define the hyperparameters we want to keep static
learner$param_set$values <- mlr3misc::insert_named(
  learner$param_set$values,
    list(
    "boosting" = "dart",
    "seed" = 17L,
    "num_iterations" = 50,
    "objective" = "regression",
    "convert_categorical" = TRUE
  )
)

tuner <- tnr("mbo")


# create a tuning instance for our tuning
instance <- ti(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 5),
  measures = msr("regr.rmse"),
  terminator = trm("stagnation_batch", n = 1, threshold = 0.5)
)


# now we are ready to tune our model
tuner$optimize(instance)


# assign tuned hyperparameters
best_bagging_fraction = instance$result$bagging_fraction
best_num_leaves = instance$result$num_leaves
best_max_depth = instance$result$max_depth
best_learning_rate = instance$result$learning_rate
best_drop_rate = instance$result$drop_rate
best_skip_drop = instance$result$skip_drop


# Training the model
# initialize the learner
best_learner <- mlr3::lrn(
  "regr.lightgbm"
)


# insert tuned hyperparameter
best_learner$param_set$values <- mlr3misc::insert_named(
  best_learner$param_set$values,
    list(
      "bagging_fraction" = best_bagging_fraction,
      "num_leaves" = best_num_leaves,
      "max_depth" = best_max_depth,
      "learning_rate" = best_learning_rate,
      "drop_rate" = best_drop_rate,
      "skip_drop" = best_skip_drop,
      "boosting" = "dart",
      "seed" = 17L,
      "num_iterations" = 100,
      "objective" = "regression",
      "convert_categorical" = TRUE
  )
)


# fit the model on the training data
lgbm_model <- best_learner$train(task)


# check importance of features for the fitted model
lgbm_importance <- lgbm_model$importance()
lgbm_importance

# lag7d        lag1d        item         month        weekday      store        running_days  day         is_weekend   seasonal 
# 0.8956642954 0.0474605233 0.0147734597 0.0101244545 0.0084737464 0.0084097696 0.0073982923 0.0051852725 0.0016367143 0.0006645874 
# day_of_month 
# 0.0002088846  


# Checking model performance on train data
fitted <- lgbm_model |>
  predict(train_df2)

final_df2 <- cbind(train_df2, y_train, fitted)

train_mape <- final_df2 |>
  as.data.frame() |>
  ungroup() |>
  mutate(MAPE_Inv = (y_train - fitted) / y_train) |>
  mutate(MAPE_Inv_abs = abs(MAPE_Inv))

train_mape <- train_mape |>
  arrange(store, item)


# summarize the results for each combination of categories
summary_train <- train_mape |>
  group_by(store, item) |>
  summarize(Average_MAPE_Train = 1 - mean(MAPE_Inv_abs))

summary_train <- summary_train[!is.na(summary_train$Average_MAPE_Train), ]
summary_train <- summary_train[summary_train$Average_MAPE_Train <= 1, ]

summary_train


# Checking model performance on test data
# initialize final predicted dataframe, and copy test_df2
predicted_df <- data.frame()
temp_df <- test_df2


# we first do the forecast for first 7 days since their lag7d are observed
for (j in 1:7) {
  for (k in as.vector(unique(test_df$store))) {
    for (l in as.vector(unique(test_df$item))) {
      forecast_df <-
        filter(temp_df, store == k, item == l, day == j, month == 1)

      forecast_sale <- lgbm_model |>
        predict(forecast_df)
      
      forecast_df <- forecast_df |>
        cbind(forecast_sale)
      
      predicted_df <- predicted_df |>
        rbind(forecast_df)
    }
  }
  # updating lag1d for next period
  forecast_df <-
    filter(temp_df, day == j + 1, month == 1)
  
  forecast_df$lag1d <-
    predicted_df$forecast_sale[(nrow(predicted_df) - ((length(
      unique(test_df$item)
    ) * length(
      unique(test_df$store)
    )) - 1)):nrow(predicted_df)]
  
  merged_df <-
    merge(
      temp_df,
      forecast_df,
      by = c("store", "item", "month", "day"),
      all.x = TRUE
    )
  
  temp_df$lag1d <-
    ifelse(!is.na(merged_df$lag1d.y),
           merged_df$lag1d.y,
           temp_df$lag1d)
}


# now we do for the rest of the days in the first month
for (i in 1) {
  for (j in 8:length(unique(filter(test_df, month == i)$day))) {
    # updating lag1d for next period
    forecast_df <-
      filter(temp_df, day == j, month == i)
    
    forecast_df$lag1d <-
      predicted_df$forecast_sale[(nrow(predicted_df) - ((length(
        unique(test_df$item)
      ) * length(
        unique(test_df$store)
      )) - 1)):nrow(predicted_df)]
    
    # updating lag7d for next period
    forecast_df$lag7d <-
      predicted_df$forecast_sale[(nrow(predicted_df) - (((length(unique(test_df$item)) * length(unique(test_df$store))) * 7) - 1)):(nrow(predicted_df) - ((length(unique(test_df$item)) * length(unique(test_df$store))) * 6))]
    
    merged_df <-
      merge(
        temp_df,
        forecast_df,
        by = c("store", "item", "month", "day"),
        all.x = TRUE
      )
    
    temp_df$lag1d <-
      ifelse(!is.na(merged_df$lag1d.y),
             merged_df$lag1d.y,
             temp_df$lag1d)
    
    temp_df$lag7d <-
      ifelse(!is.na(merged_df$lag7d.y),
             merged_df$lag7d.y,
             temp_df$lag7d)
    
    for (k in as.vector(unique(test_df$store))) {
      for (l in as.vector(unique(test_df$item))) {
        forecast_df <-
          filter(temp_df, store == k, item == l, day == j, month == i)

        forecast_sale <- lgbm_model |>
          predict(forecast_df)
        
        forecast_df <- forecast_df |>
          cbind(forecast_sale)
        
        predicted_df <- predicted_df |>
          rbind(forecast_df)
      }
    }
  }
}


# now we do for the rest of the days for the rest of the months
for (i in 2:12) {
  for (j in 1:length(unique(filter(test_df, month == i)$day))) {
    # updating lag1d for next period
    forecast_df <-
      filter(temp_df, day == j, month == i)
    
    forecast_df$lag1d <-
      predicted_df$forecast_sale[(nrow(predicted_df) - ((length(
        unique(test_df$item)
      ) * length(
        unique(test_df$store)
      )) - 1)):nrow(predicted_df)]
    
    # updating lag7d for next period
    forecast_df$lag7d <-
      predicted_df$forecast_sale[(nrow(predicted_df) - (((length(unique(test_df$item)) * length(unique(test_df$store))) * 7) - 1)):(nrow(predicted_df) - ((length(unique(test_df$item)) * length(unique(test_df$store))) * 6))]
    
    merged_df <-
      merge(
        temp_df,
        forecast_df,
        by = c("store", "item", "month", "day"),
        all.x = TRUE
      )
    
    temp_df$lag1d <-
      ifelse(!is.na(merged_df$lag1d.y),
             merged_df$lag1d.y,
             temp_df$lag1d)
    
    temp_df$lag7d <-
      ifelse(!is.na(merged_df$lag7d.y),
             merged_df$lag7d.y,
             temp_df$lag7d)
    
    for (k in as.vector(unique(test_df$store))) {
      for (l in as.vector(unique(test_df$item))) {
        forecast_df <-
          filter(temp_df, store == k, item == l, day == j, month == i)

        forecast_sale <- lgbm_model |>
          predict(forecast_df)
        
        forecast_df <- forecast_df |>
          cbind(forecast_sale)
        
        predicted_df <- predicted_df |>
          rbind(forecast_df)
      }
    }
  }
}

final_test_df2 <- cbind(predicted_df |>
                          arrange(store, item, month, day), y_test)

test_mape <- final_test_df2 |>
  as.data.frame() |>
  ungroup() |>
  mutate(MAPE_Inv = (y_test - forecast_sale) / y_test) |>
  mutate(MAPE_Inv_abs = abs(MAPE_Inv))

test_mape <- test_mape |>
  arrange(store, item)


# summarize the results for each combination of categories
summary_test <- test_mape |>
  group_by(store, item) |>
  summarize(Average_MAPE_Test = 1 - mean(MAPE_Inv_abs))

summary_test <- summary_test[!is.na(summary_test$Average_MAPE_Test), ]
summary_test <- summary_test[summary_test$Average_MAPE_Test <= 1, ]

summary_test


# Plot to show predicted sales vs actual sales figure for test set
forecast_performance <- final_test_df2 |>
  filter(store == "1", item == "1") |>
  mutate(year = "2017")

forecast_performance$date <- as.Date(paste(forecast_performance$year, forecast_performance$month, forecast_performance$day, sep = "-"))

forecast_analysis <- ggplot() +
  geom_line(data = forecast_performance, aes(x = as.Date(date), y = y_test, color = "Actual"), linetype = "dotted") +
  geom_line(data = forecast_performance, aes(x = as.Date(date), y = forecast_sale, color = "Predicted")) +
  scale_x_date(labels = date_format("%Y-%b"), breaks = date_breaks("1 months")) +
  xlab("Date") +
  ylab("Amount ($)") +
  ggtitle("Forecasted Sales for Store 1, Item 1") +
  scale_color_manual(values = c("Actual" = "#009E73", "Predicted" = "#CC79A7"), name = "Legend") +
  theme_economist()

ggplotly(forecast_analysis, viewer = "new")


# Compare train and test accuracy
summary_all <- summary_train |>
  left_join(summary_test, by = c('store', 'item')) |>
  mutate(accuracy_gap = round(Average_MAPE_Train - Average_MAPE_Test , 2)) |>
  arrange(desc(accuracy_gap))

summary_all


# Predict seasonality for next year
# initialize future dates and dataframe
date_vector <- seq(as.Date("2018-01-01"), as.Date("2018-12-31"), by = "day")


# create a weighted mean for seasonality using simple exponential smoothing for future days for a year based on past 5 years
forecast_seasonal <- c()

for(i in as.vector(unique(test_df$store))) {
  for (j in as.vector(unique(test_df$item))) {
    for (k in 1:length(date_vector)) {
      filtered_df <- df_w_seas |>
        filter(
          store == i,
          item == j,
          as.integer(month) == month(date_vector[k]),
          as.integer(day) == day(date_vector[k])
        )
      
      decay_level <-
        0.2 # choose the appropriate decay factor alpha where 0<=r<=1
      weighted_mean_seasonality <-
        forecast::ses(filtered_df$seasonal, alpha = decay_level, h = 1)
      
      forecast_seasonal <-
        append(forecast_seasonal, weighted_mean_seasonality$x[1])
    }
  }
}

store_vector <- rep(as.vector(unique(train_df$store)), each = length(date_vector) * length(as.vector(unique(train_df$item))))
item_vector <- rep(as.vector(unique(train_df$item)), each = length(date_vector), times = length(as.vector(unique(train_df$store))))

# create a dataframe to store forecasted seasonality for each store and item at each day
next_forecast <- data.frame(
  date = rep(date_vector, length(as.vector(
    unique(train_df$store)
  )) * length(as.vector(
    unique(train_df$item)
  ))),
  store = store_vector,
  item = item_vector,
  seasonal = forecast_seasonal
)


# Predicting new unseen data
# adding in required variables
next_forecast$month <- as.factor(month(next_forecast$date))
next_forecast$day <- as.factor(day(next_forecast$date))
next_forecast$store <- factor(next_forecast$store, levels = levels(train_df2$store))
next_forecast$item <- factor(next_forecast$item, levels = levels(train_df2$item))
next_forecast$lag1d <- rep(NA_real_, nrow(next_forecast))
next_forecast$lag7d <- rep(NA_real_, nrow(next_forecast))

next_forecast <- next_forecast |>
  arrange(store, item, date) |>
  group_by(store, item) |>
  mutate(running_days = row_number() + max(final_test_df2$running_days)) |>  
  mutate(day_of_month = case_when(
    day(date) == 1 ~ "start",
    day(date) == days_in_month(date) ~ "end",
    TRUE ~ "normal"
    )) |>
  ungroup() |>
  cbind(data.frame(weekday = factor(wday(next_forecast$date, label=TRUE), ordered = FALSE))) |>
  mutate(is_weekend = case_when(
    weekday %in% c("Sat", "Sun") ~ "yes",
    TRUE ~ "no"
  )) |>
  mutate_if(is.character, as.factor) |>
  select(c(date, seasonal, lag1d, lag7d, store, item, month, day, day_of_month, weekday, is_weekend, running_days))

df_w_seas_filtered$date <- df_w_seas_filtered$date + days(1)

lag1d <- df_w_seas_filtered[df_w_seas_filtered$date == as.Date("2018-01-01"), ]

merged_df <-
  merge(next_forecast,
        lag1d,
        by = c("store", "item", "date"),
        all.x = TRUE)

next_forecast$lag1d <-
  ifelse(!is.na(merged_df$sales),
         merged_df$sales,
         next_forecast$lag1d)

df_w_seas_filtered$date <- df_w_seas_filtered$date + days(6)

lag7d <- df_w_seas_filtered[df_w_seas_filtered$date >= as.Date("2018-01-01") & df_w_seas_filtered$date <= as.Date("2018-01-07"), ]

merged_df <-
  merge(next_forecast,
        lag7d,
        by = c("store", "item", "date"),
        all.x = TRUE)

next_forecast$lag7d <-
  ifelse(!is.na(merged_df$sales),
         merged_df$sales,
         next_forecast$lag7d)

next_forecast <- next_forecast |>
  select(- date)


# initialize final predicted dataframe, and copy next_forecast
final_forecast <- data.frame()
temp_df2 <- next_forecast


# we first do the forecast for first 7 days since their lag7d are observed
for (j in 1:7) {
  for (k in as.vector(unique(next_forecast$store))) {
    for (l in as.vector(unique(next_forecast$item))) {
      forecast_df <-
        filter(temp_df2, store == k, item == l, day == j, month == 1)

      forecast_sale <- lgbm_model |>
        predict(forecast_df)
      
      forecast_df <- forecast_df |>
        cbind(forecast_sale = forecast_sale)
      
      final_forecast <- final_forecast |>
        rbind(forecast_df)
    }
  }
  # updating lag1d for next period
  forecast_df <-
    filter(temp_df2, day == j + 1, month == 1)
  
  forecast_df$lag1d <-
    final_forecast$forecast_sale[(nrow(final_forecast) - ((length(
      unique(next_forecast$item)
    ) * length(
      unique(next_forecast$store)
    )) - 1)):nrow(final_forecast)]
  
  merged_df <-
    merge(
      temp_df2,
      forecast_df,
      by = c("store", "item", "month", "day"),
      all.x = TRUE
    )
  
  temp_df2$lag1d <-
    ifelse(!is.na(merged_df$lag1d.y),
           merged_df$lag1d.y,
           temp_df2$lag1d)
}


# now we do for the rest of the days in the first month
for (i in 1) {
  for (j in 8:length(unique(filter(next_forecast, month == i)$day))) {
    # updating lag1d for next period
    forecast_df <-
      filter(temp_df2, day == j, month == i)
    
    forecast_df$lag1d <-
      final_forecast$forecast_sale[(nrow(final_forecast) - ((length(
        unique(next_forecast$item)
      ) * length(
        unique(next_forecast$store)
      )) - 1)):nrow(final_forecast)]
    
    # updating lag7d for next period
    forecast_df$lag7d <-
      final_forecast$forecast_sale[(nrow(final_forecast) - (((length(unique(next_forecast$item)) * length(unique(next_forecast$store))) * 7) - 1)):(nrow(final_forecast) - ((length(unique(next_forecast$item)) * length(unique(next_forecast$store))) * 6))]
    
    merged_df <-
      merge(
        temp_df2,
        forecast_df,
        by = c("store", "item", "month", "day"),
        all.x = TRUE
      )
    
    temp_df2$lag1d <-
      ifelse(!is.na(merged_df$lag1d.y),
             merged_df$lag1d.y,
             temp_df2$lag1d)
    
    temp_df2$lag7d <-
      ifelse(!is.na(merged_df$lag7d.y),
             merged_df$lag7d.y,
             temp_df2$lag7d)
    
    for (k in as.vector(unique(next_forecast$store))) {
      for (l in as.vector(unique(next_forecast$item))) {
        forecast_df <-
          filter(temp_df2, store == k, item == l, day == j, month == i)

        forecast_sale <- lgbm_model |>
          predict(forecast_df)
        
        forecast_df <- forecast_df |>
          cbind(forecast_sale = forecast_sale)
        
        final_forecast <- final_forecast |>
          rbind(forecast_df)
      }
    }
  }
}


# now we do for the rest of the days for the rest of the months
for (i in 2:12) {
  for (j in 1:length(unique(filter(next_forecast, month == i)$day))) {
    # updating lag1d for next period
    forecast_df <-
      filter(temp_df2, day == j, month == i)
    
    forecast_df$lag1d <-
      final_forecast$forecast_sale[(nrow(final_forecast) - ((length(
        unique(next_forecast$item)
      ) * length(
        unique(next_forecast$store)
      )) - 1)):nrow(final_forecast)]
    
    # updating lag7d for next period
    forecast_df$lag7d <-
      final_forecast$forecast_sale[(nrow(final_forecast) - (((length(unique(next_forecast$item)) * length(unique(next_forecast$store))) * 7) - 1)):(nrow(final_forecast) - ((length(unique(next_forecast$item)) * length(unique(next_forecast$store))) * 6))]
    
    merged_df <-
      merge(
        temp_df2,
        forecast_df,
        by = c("store", "item", "month", "day"),
        all.x = TRUE
      )
    
    temp_df2$lag1d <-
      ifelse(!is.na(merged_df$lag1d.y),
             merged_df$lag1d.y,
             temp_df2$lag1d)
    
    temp_df2$lag7d <-
      ifelse(!is.na(merged_df$lag7d.y),
             merged_df$lag7d.y,
             temp_df2$lag7d)
    
    for (k in as.vector(unique(next_forecast$store))) {
      for (l in as.vector(unique(next_forecast$item))) {
        forecast_df <-
          filter(temp_df2, store == k, item == l, day == j, month == i)

        forecast_sale <- lgbm_model |>
          predict(forecast_df)
        
        forecast_df <- forecast_df |>
          cbind(forecast_sale = forecast_sale)
        
        final_forecast <- final_forecast |>
          rbind(forecast_df)
      }
    }
  }
}


# final forecast dataframe with forecasted sales for all groups for 1 year
View(final_forecast)


# Plot to show predicted sales
forecast_future <- final_forecast |>
  filter(store == "1", item == "1") |>
  mutate(year = "2018")

forecast_future$date <- as.Date(paste(forecast_future$year, forecast_future$month, forecast_future$day, sep = "-"))

forecast_future_analysis <- ggplot() +
  geom_line(data = forecast_future, aes(x = as.Date(date), y = forecast_sale, color = "Predicted")) +
  scale_x_date(labels = date_format("%Y-%b"), breaks = date_breaks("1 months")) +
  xlab("Date") +
  ylab("Amount ($)") +
  ggtitle("Forecasted Sales for Store 1, Item 1") +
  scale_color_manual(values = c("Predicted" = "#009E73"), name = "Legend") +
  theme_economist()

ggplotly(forecast_future_analysis, viewer = "new")

