## Avaliação do Modelo Colaborativo

O algoritmo SVD foi avaliado utilizando validação cruzada com 5 folds.

Resultados obtidos:

- RMSE: 0.9522
- MAE: 0.7530

Os resultados indicam que o modelo apresenta erro médio inferior a uma estrela em uma escala de avaliação de 1 a 5.

Além disso, o baixo desvio padrão entre os folds demonstra estabilidade e capacidade de generalização, indicando que o modelo não apresenta sinais significativos de overfitting.

### Análise de Atividade dos Usuários

A análise do histórico de interações revelou uma elevada esparsidade na matriz usuário-item.

Dos 14.266 usuários presentes no conjunto de dados, 85,65% possuem no máximo cinco avaliações registradas. Além disso, apenas 0,62% dos usuários apresentam mais de vinte avaliações.

Esse cenário limita significativamente a capacidade dos algoritmos colaborativos de identificar padrões robustos de similaridade entre usuários, uma vez que a maior parte da base não possui histórico suficiente para modelagem de preferências.

Tal característica explica o ganho modesto obtido pelo modelo SVD em relação ao baseline de popularidade e justifica a adoção de uma arquitetura híbrida que combine filtragem colaborativa, filtragem baseada em conteúdo e estratégias para tratamento de cold start.
