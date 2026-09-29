# syntax=docker/dockerfile:1
# Фронтенд: сборка Vite и раздача nginx; /api/ nginx проксирует в приложение (docker/frontend.nginx.conf).
# Собирается из корня репозитория: docker compose build frontend

FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY frontend/ ./
# mock — демонстрационные данные без бэкенда; http — запросы в VITE_API_BASE_URL
ARG VITE_API_MODE=http
ARG VITE_API_BASE_URL=/api
ENV VITE_API_MODE=$VITE_API_MODE \
    VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:1.30-alpine
COPY docker/frontend.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
