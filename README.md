# 📚 Book Recommendation API

Sistema de recomendação de livros desenvolvido com Python e FastAPI utilizando técnicas de **Content-Based Filtering**, **Collaborative Filtering** e **Recomendação Híbrida Adaptativa**.

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como atividade acadêmica com o objetivo de construir uma API capaz de recomendar livros a partir de diferentes estratégias de recomendação.

O sistema utiliza dados de livros e avaliações de usuários para gerar recomendações personalizadas através de três abordagens:

* Content-Based Filtering (baseado em conteúdo)
* Collaborative Filtering (baseado em usuários semelhantes)
* Sistema Híbrido Adaptativo

Além disso, foi realizada a avaliação quantitativa dos modelos utilizando métricas clássicas de sistemas de recomendação.

---

# 🎯 Objetivos

* Construir uma API REST para recomendação de livros.
* Implementar diferentes algoritmos de recomendação.
* Comparar desempenho entre abordagens.
* Disponibilizar endpoints documentados via Swagger.
* Containerizar a aplicação utilizando Docker.

---

# 🛠 Tecnologias Utilizadas

## Backend

* Python 3.12
* FastAPI
* Uvicorn

## Manipulação de Dados

* Pandas
* NumPy

## Machine Learning

* Scikit-Learn
* Surprise

## Testes

* Pytest

## Containerização

* Docker
* Docker Compose

---

# 📂 Dataset

O projeto utiliza dois conjuntos de dados:

### Books

Informações dos livros:

* bookId
* title
* author
* rating
* likedPercent
* genres
* description

### Ratings

Informações de avaliações:

* user_id
* book_id
* rating
* is_read
* is_reviewed

Após o processo de limpeza:

* 52.428 livros
* 46.631 avaliações
* 14.266 usuários únicos

---

# 🧠 Modelos Implementados

## 1. Content-Based Filtering

Recomenda livros similares com base no conteúdo textual.

### Features Utilizadas

* Título
* Autor
* Gêneros
* Descrição

### Técnica

TF-IDF + Similaridade do Cosseno

Fluxo:

Livro → Vetorização TF-IDF → Similaridade → Recomendações

---

## 2. Collaborative Filtering

Recomenda livros com base em padrões de usuários semelhantes.

### Algoritmo

SVD (Singular Value Decomposition)

Biblioteca:

```python
surprise.SVD
```

O modelo aprende fatores latentes entre usuários e livros para prever avaliações futuras.

---

## 3. Sistema Híbrido

Combina:

* Content-Based
* Collaborative Filtering

As recomendações são unificadas em uma única lista contendo:

* título
* origem da recomendação
* justificativa
* score

Exemplo:

```json
{
  "title": "American Dirt",
  "source": "content",
  "reason": "Similar ao livro favorito",
  "score": 0.2866
}
```

---

## 4. Sistema Adaptativo

O algoritmo escolhe automaticamente a melhor estratégia para cada usuário.

### Poucas avaliações (< 5)

Estratégia:

Popularity Based

### Usuários intermediários (5–20)

Estratégia:

Hybrid Recommendation

### Usuários ativos (> 20)

Estratégia:

Collaborative Filtering

---

# 📊 Avaliação dos Modelos

## Collaborative Filtering (SVD)

Validação cruzada com 5 folds.

### Resultados

RMSE:

```text
0.9522
```

MAE:

```text
0.7530
```

---

## Baseline

Predição pela média global das avaliações.

RMSE:

```text
0.9760
```

MAE:

```text
0.7565
```

---

## Comparação

| Modelo   | RMSE   | MAE    |
| -------- | ------ | ------ |
| Baseline | 0.9760 | 0.7565 |
| SVD      | 0.9522 | 0.7530 |

O modelo SVD apresentou desempenho superior ao baseline.

---

# 🚀 Executando o Projeto

## Clonar o Repositório

```bash
git clone <url-do-repositorio>

cd projeto_sistema_recomendacao_api
```

---

## Criar Ambiente Virtual

Linux:

```bash
python -m venv .venv

source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

## Instalar Dependências

O projeto utiliza o gerenciador de pacotes **uv**.

Instalar as dependências:

```bash
uv sync
```

---

## Executar API

```bash
uvicorn app.main:app --reload
```

Aplicação:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

# 🐳 Docker

Build:

```bash
docker compose build
```

Subir aplicação:

```bash
docker compose up
```

Swagger:

```text
http://localhost:8000/docs
```

---

# 🔗 Endpoints

## Health Check

```http
GET /health
```

---

## Informações do Dataset

```http
GET /dataset/stats
```

---

## Recomendação por Conteúdo

```http
GET /recommend/content/{title}
```

Exemplo:

```http
GET /recommend/content/The Hunger Games
```

---

## Recomendação Colaborativa

```http
GET /recommend/collaborative/{user_id}
```

Exemplo:

```http
GET /recommend/collaborative/9880
```

---

## Recomendação Híbrida

```http
GET /recommend/hybrid/{user_id}
```

Exemplo:

```http
GET /recommend/hybrid/9880
```

---

# 🧪 Testes

Executar todos os testes:

```bash
pytest tests/
```

Testes implementados:

* Carregamento dos datasets
* Content-Based Filtering
* Collaborative Filtering
* Sistema Híbrido

Resultado esperado:

```text
5 passed
```

---

# 📁 Estrutura do Projeto

```text
.
├── app/
├── tests/
├── scripts/
├── data/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

# 👨‍💻 Autor
Lucas Petry
Projeto desenvolvido para fins acadêmicos 
