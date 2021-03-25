source("Dispensers/R/GetCleanTime.R")
library(tidyverse)
library(ForecastHelpers)

theme_set(theme_minimal())
current_month <- '2021-03-01'
cutoff <- as.Date('2020-08-01')
training_month <- as.Date('2020-10-01')

con <- DBI::dbConnect(odbc::odbc(),
                      Driver = "SQL Server",
                      Server = "192.168.66.59",
                      Database = "Data Warehouse",
                      Port = 1433
)

item_list <- c("601090",	"601132",	"900114",	"900127",	"601225-B",	"601205-C",	"601087-C",	"601088-C",	"601090-C",	"601090 SO-C",	"601130-C",	"601132-C",	"601148",	"601240-B",	"601243-C",	"601244",	"601256-C",	"601271-C",	"900127-C",	"900130-C",	"900196",	"900179",	"900177",	"601088",	"601234-C",	"601087",	"601200",	"601202",	"601203",	"601130",	"900130",	"601142-C",	"601144-C",	"601229",	"601141",	"601142",	"601144",	"601205",	"601158",	"601159",	"601160",	"601242",	"601243",	"601292",	"601321-C",	"601278-C",	"601272-C",	"601242-C",	"601258-C",	"601264-C",	"601305-C",	"601324-C",	"601258",	"601323",	"601325",	"601165",	"601204",	"601166",	"601167",	"601178",	"601137",	"601256 SO-C",	"601323-C",	"601275",	"601275-C",	"601171",	"601177",	"601177-C",	"601231",	"601231-C",	"601234",	"601236",	"601256",	"601259",	"601264",	"601272",	"601281",	"601282",	"601283",	"601285",	"601286",	"601287",	"601288")

sales_history <- tbl(con, "PWV_GPSalesHistory_PRMW")

top_customers <- c(
  "Walmart",
  "Lowes Home Improvement",
  "Walmart-Di",
  "Sam's Club",
  "Homedepot-Prod",
  "Walmart-Di-Cad",
  "Costco",
  "Walmartc",
  "Amazon_sf",
  "Sams.com",
  "Z-9001",
  "Z-9000",
  "Homedepot.com",
  "Bed Bath & Beyond",
  "Samdsv",
  "Wayfair",
  "Target",
  "Do It Best",
  "Bimartprod",
  "Ace Hardware Stores"
)


dispenser_items <- tbl(con, 'PWV_GPSalesHistory_PRMW') %>% 
  filter(ITEMNMBR %in% item_list & year(Glpostdt) >= 2020) %>% 
  select(ITEMNMBR, ITEMDESC, ItmClsCd, RptClass, Brand) %>% 
  collect() %>% 
  mutate(across(where(is.character), trimws)) %>% 
  distinct(ITEMNMBR, .keep_all = TRUE)


data <- sales_history %>%
  filter(ITEMNMBR %in% item_list & Year >= 2017 & SopType != "Return" & Glpostdt < current_month) %>%
  collect() %>%
  janitor::clean_names("upper_camel")

data_cust <- data %>%
  mutate(
    across(where(is.character), stringr::str_squish),
    ChainName = str_to_title(ChainName),
    Custname = str_to_title(Custname),
    Custname = if_else(ChainName %in% top_customers, Custname, "All Other"),
    Custname = case_when(grepl("Wayfair", Custname) ~ 'Wayfair Llc',
                         Custname %in% c('Walmart', 'Wal-Mart') ~ 'Walmart',
                         Custname %in% c('Target.com','Target .Com') ~ 'Target.com',
                         Custname == "Sam' S Club" ~ 'Sams Club',
                         Custname == 'PRIMO WATER COM' ~ 'Primowater.com',
                         Custname == 'Sams Dsv' ~ 'Sams Drop Shipment',
                         Custname %in% c('Lowes', 'Lowes Home Improvement','Lowes Home Improvement - Whse') ~ 'Lowes',
                         grepl('Do It Best', Custname) ~ 'Do It Best',
                         Custname == 'David Macias' ~ 'All Other',
                         Custname %in% c('Costcoc', 'Costco.com', 'Costco .Com') ~ 'Costco.com',
                         grepl('Ace Hardware', Custname) ~ 'Ace Hardware',
                         Custname %in% c('Bed Bath Beyond .Com', 'Bed Bath .Com') ~ 'BedBathBeyond.com',
                         Custname == 'Lowes .Com' ~ 'Lowes.com',
                         TRUE ~ Custname)
    )

top_items <- data_cust %>% 
  filter(lubridate::year(Glpostdt) == 2021) %>% 
  group_by(Itemnmbr) %>% 
  summarise(Quantity = sum(Quantity,na.rm = TRUE)) %>% 
  arrange(desc(Quantity))

top_10_items <- top_items %>% 
  slice(1:10)

#this works better
top_items_cust <- data_cust %>% 
  filter(lubridate::year(Glpostdt) == 2021) %>% 
  group_by(Custname, Itemnmbr) %>% 
  summarise(Quantity = sum(Quantity,na.rm = TRUE)) %>% 
  arrange(desc(Quantity)) %>% 
  slice_max(n=2, with_ties = F, order_by= Quantity) %>% 
  mutate(ID = paste0(Custname,Itemnmbr))


last_trans_date <- data_cust %>% 
  group_by(Itemnmbr, Itemdesc, Custname) %>% 
  summarise(Date = max(Glpostdt))

last_trans_date_clean <- last_trans_date %>% 
  filter(Date >= cutoff) %>% 
  mutate(ID = paste0(Custname, Itemnmbr))

data_month <- get_clean_time(data_cust, 'month')
data_week <- get_clean_time(data_cust, 'week')

write_csv(data_month,'Dispensers/Data/DispenserMonth.csv')
write_csv(data_week,'Dispensers/Data/DispenserWeek.csv')