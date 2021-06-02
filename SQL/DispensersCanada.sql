/****** Script for SelectTopNRows command from SSMS  ******/
 SELECT *
 INTO #temp
 FROM [Data Warehouse].dbo.cfg_Customer
 WHERE [Country] LIKE '%CAN%' AND NOT [State] IN ('NC', 'NJ')



SELECT *
  FROM [Data Warehouse].[dbo].[PWV_GPSalesHistory_PRMW] as t
  WHERE t.State IN
	(SELECT DISTINCT [#temp].State
	FROM #temp
	) AND
	t.[RptClass] LIKE '%DISPEN%' AND 
  [Year] >= 2020

DROP TABLE #temp



SELECT [CUSTNAME], [ChainName], [Year], SUM(QUANTITY) AS 'QTY'
  FROM [Data Warehouse].[dbo].[PWV_GPSalesHistory_PRCN] as t
  WHERE [RptClass] LIKE '%DISPEN%' AND [GLPOSTDT] >= '2016-01-01' AND [SopType] = 'Invoice'
  GROUP BY [CUSTNAME], [Year] , [ChainName]
  ORDER BY [Year]


SELECT DISTINCT [ITEMNMBR], [ITEMDESC], [RptClass]
  FROM [Data Warehouse].[dbo].[PWV_GPSalesHistory_PRCN] as t
  WHERE [ITEMNMBR] IN ('860011-C',
'860015-C',
'860019',
'860021',
'860025-C',
'860027-C',
'860010',
'860012',
'860013',
'860014-C',
'860017-C',
'860013-C',
'860011',
'860016-C')
