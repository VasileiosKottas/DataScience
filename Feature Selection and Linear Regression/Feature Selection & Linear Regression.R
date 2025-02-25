library(dplyr)
data <- read.csv(file.choose())

# a
house_prices_numeric <- data %>% select(where(is.numeric))
house_prices_numeric <- scale(house_prices_numeric)
cor_matrix <- cor(house_prices_numeric)
print(cor_matrix)

# b
cor_price <- cor_matrix["price", ]
sorted_features <- sort(cor_price, decreasing = TRUE)

selected_features <- names(sorted_features)[2:4]

print(selected_features)

barplot(sorted_features[2:4], names.arg = selected_features, col = "blue",
        main = "Top 3 Features Correlated with Price",
        ylab = "Correlation Coefficient")

# c
formula <- as.formula(paste("price ~", paste(selected_features, collapse = " + ")))
model <- lm(formula, data = as.data.frame(house_prices_numeric))

r_squared <- summary(model)$r.squared
print(paste("R-squared value:", r_squared))

predictions <- predict(model, newdata = as.data.frame(house_prices_numeric))
data_top3$price <- house_prices_numeric[, "price"]
mae <- mean(abs(predictions - actuals))
print(paste("Mean Absolute Error:", mae))

# d

price_mean <- mean(data$price, na.rm = TRUE)
price_sd <- sd(data$price, na.rm = TRUE)

denormalized_predictions <- (predictions * price_sd) + price_mean
denormalized_actual <- (data_top3$price * price_sd) + price_mean

specific_ids <- c(5, 100, 305)
specific_denormalized_predictions <- round(denormalized_predictions[specific_ids], 3)
print(specific_denormalized_predictions)

plot(denormalized_actual, denormalized_predictions, col = "blue", pch = 16,
     main = "Actual vs Predicted House Prices",
     xlab = "Actual Prices", ylab = "Predicted Prices")
abline(0, 1, col = "red", lwd = 2)