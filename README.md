# Databricks - Demand Forecasting Template
## Situação atual

Um grande e-commerce internacional está observando problemas relacionadas a sua gestão de estoque. Perde-se oportunidade de vender produtos que estão sem estoque e há produtos com muito estoque e sem vendas há longos períodos.

## Objetivo

Otimizar o estoque de um e-commerce com base na demanda real de produtos.

## Dados

bigquery-public-data.thelook_ecommerce.events

## Premissas

As compras são planejadas bimestralmente

Todos os itens tem mais ou menos o mesmo leadtime de 1 semana

Estoque de segurança é a quantidade para suprir a demanda de 1 semana

Desconsiderando pedidos cancelados

## Entregáveis

- Prever a demanda de cada centro de distribuição nas próximas 2 semanas (em unidades vendidas)
- Assumindo que cada o de mix de produtos de cada centro de distribuição muda a cada três meses, é possível calcular a quantidade em demanda de cada produto e os items que o compõe