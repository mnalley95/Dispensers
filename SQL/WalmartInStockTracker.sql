/****** Script for SelectTopNRows command from SSMS  ******/
SELECT [Retailer], [PrimoItem], [ItemDesc], [ItemType], [WeekEnding], SUM(InStock)/COUNT(InStock) AS 'In Stock Percentage'
  FROM [Data Warehouse].[dbo].[WSUImport]
  WHERE [WeekEnding] >= '2021-01-01' AND [Retailer] = 'WALM'
  GROUP BY [Retailer], [PrimoItem], [ItemDesc], [ItemType], [WeekEnding]
  ORDER BY [WeekEnding]