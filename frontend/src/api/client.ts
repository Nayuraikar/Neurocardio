import axios, { AxiosError } from 'axios'

export const http = axios.create({ baseURL: '/api' })

export type ApiError = { status: number | null; message: string; code?: string }

export function toApiError(e: unknown): ApiError {
  const x = e as AxiosError<{ detail?: unknown; code?: string }>
  const status = x.response?.status ?? null
  const detail = x.response?.data?.detail
  if (!x.response) return { status, message: 'Cannot reach the API. Check that the backend is running.' }
  if (typeof detail === 'string') return { status, message: detail, code: x.response.data?.code }
  if (Array.isArray(detail)) return { status, message: detail.map((d: { msg?: string }) => d.msg).join('; ') }
  if (detail && typeof detail === 'object') return { status, message: String((detail as { message?: string }).message ?? 'Request failed') }
  return { status, message: status && status >= 500 ? `The API failed (HTTP ${status}).` : `Request failed (HTTP ${status}).` }
}

export async function get<T>(url: string, params?: object): Promise<T> {
  return (await http.get<T>(url, { params })).data
}

export async function post<T>(url: string, body?: unknown, params?: object): Promise<T> {
  return (await http.post<T>(url, body, { params })).data
}
