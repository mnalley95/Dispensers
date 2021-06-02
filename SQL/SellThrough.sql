/****** Script for SelectTopNRows command from SSMS  ******/
SELECT [Chain]
      ,MONTH([TransDate]) as 'month'
	  ,SUM(QTY) AS 'Qty'
  FROM [Data Warehouse].[dbo].[WPMImport]
  WHERE [TransDate] > '2021-01-01' and [Sale_Return] = 'S'
  GROUP BY [Chain], MONTH(TransDate)
  ORDER BY [Chain], MONTH(TransDate)