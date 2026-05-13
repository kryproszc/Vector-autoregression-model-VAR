import type { AuthSession } from "@/features/auth/types";

const AUTH_SESSION_KEY = "triangle.auth.session";
const UI_SESSION_PREFIX = "triangle.ui.";

export function getStoredAuthSession(): AuthSession | null {
	if (typeof window === "undefined") {
		return null;
	}

	// Keep session only for the current browser/tab lifetime.
	const raw = window.sessionStorage.getItem(AUTH_SESSION_KEY);
	if (!raw) {
		// Cleanup legacy persisted auth from previous implementation.
		window.localStorage.removeItem(AUTH_SESSION_KEY);
		return null;
	}

	try {
		const parsed = JSON.parse(raw) as Partial<AuthSession>;
		if (
			typeof parsed.token !== "string" ||
			!parsed.token ||
			typeof parsed.expiresAt !== "string" ||
			!parsed.expiresAt ||
			!parsed.user ||
			typeof parsed.user !== "object"
		) {
			return null;
		}

		return parsed as AuthSession;
	} catch {
		return null;
	}
}

export function setStoredAuthSession(session: AuthSession) {
	if (typeof window === "undefined") {
		return;
	}

	window.sessionStorage.setItem(AUTH_SESSION_KEY, JSON.stringify(session));
}

export function clearStoredAuthSession() {
	if (typeof window === "undefined") {
		return;
	}

	window.sessionStorage.removeItem(AUTH_SESSION_KEY);
	for (let index = window.sessionStorage.length - 1; index >= 0; index -= 1) {
		const key = window.sessionStorage.key(index);
		if (key?.startsWith(UI_SESSION_PREFIX)) {
			window.sessionStorage.removeItem(key);
		}
	}
	window.localStorage.removeItem(AUTH_SESSION_KEY);
}

export function getStoredAuthToken() {
	return getStoredAuthSession()?.token ?? "";
}
