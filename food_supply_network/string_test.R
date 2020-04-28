df_products_info <- read.csv(file="/Users/jdeswardt/Documents/nmrql_product_infos.csv")
df_products <- read.csv(file="/Users/jdeswardt/Documents/nmrql_products.csv")

View(df_products_info)
View(df_products)

##Transform text to lower case
df_products$name <- tolower(df_products$name)


df_products$name_mapped <- ifelse(str_detect(string=df_products$name, pattern=" yoghurt "), "YOGHURTS",
                           ifelse(str_detect(string=df_products$name, pattern=" milk "), "MILK",
                           ifelse(str_detect(string=df_products$name, pattern=" juice "), "JUICE",
                           ifelse(str_detect(string=df_products$name, pattern=" cheese "), "CHEESE",
                           ifelse(str_detect(string=df_products$name, pattern=" blanc "), "WHITE WINE",
                           ifelse(str_detect(string=df_products$name, pattern="% "), "JUICE",
                           "UNKNOWN"))))))

View(df_products)
