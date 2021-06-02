SELECT CUSTNAME, [Month], SUM([QUANTITY])
  FROM [Data Warehouse].[dbo].[PWV_GPSalesHistory_PRMW] as t
  WHERE GLPOSTDT >= '2021-01-01' AND [CUSTNAME] IN ('WAL-MART', 'HOME DEPOT', 'WALMART') AND [RptClass] LIKE '%DISPEN%' AND [SopType] = 'Invoice' AND [UNITPRCE] > 0
  GROUP BY [CUSTNAME], [Month]
  ORDER BY CUSTNAME, [Month]



SELECT DISTINCT [CUSTNAME], [GLPOSTDT], [Month], [SOPNUMBE], [QUANTITY] INTO #temp
  FROM [Data Warehouse].[dbo].[PWV_GPSalesHistory_PRMW] as t
  WHERE GLPOSTDT >= '2021-01-01' AND [CUSTNAME] IN ('WAL-MART', 'HOME DEPOT', 'WALMART') AND [RptClass] LIKE '%DISPEN%' AND [SopType] = 'Invoice'

SELECT [CUSTNAME], [Month], SUM(QUANTITY)
	FROM #temp 
	GROUP BY [CUSTNAME], [Month]
	ORDER BY CUSTNAME, [Month]

DROP TABLE #temp
