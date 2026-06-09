import type { DictionaryEntry } from "@/features/dictionaries/types";
import {
	ALL_INSPECTION_COLUMN_KEYS,
	PODSTAWOWE_INSPECTION_COLUMN_KEYS,
	TERMINY_INSPECTION_COLUMN_KEYS,
} from "@/features/inspections/data";
import type {
	InspectionColumnKey,
	InspectionRow,
	InspectionViewId,
} from "@/features/inspections/types";
import { toDateInputValue, toDateList } from "@/shared/utils/date";

export type AddInspectionForm = Omit<InspectionRow, "id" | "lp" | "kodInspekcji"> & {
	brakDataDoreczeniaPisma: boolean;
	brakDataPismaZastrzezenia: boolean;
	brakDataWyslaniaPismaZZastrzezeniami: boolean;
	brakDataWplywuPisma: boolean;
	brakDataPismaZOdpowiedzia: boolean;
	brakDataWyslaniaPismaZOdpowiedzia: boolean;
};

export type InspectionListResponse = {
	items: Array<
		InspectionRow & {
			osobaKierujacaUserId?: number;
			teamMemberUserIds?: number[];
			dataAkceptacjiNotyList?: string[];
			dataZalecenList?: string[];
			brakDataDoreczeniaPisma?: boolean;
			brakDataPismaZastrzezenia?: boolean;
			brakDataWyslaniaPismaZZastrzezeniami?: boolean;
			brakDataWplywuPisma?: boolean;
			brakDataPismaZOdpowiedzia?: boolean;
			brakDataWyslaniaPismaZOdpowiedzia?: boolean;
		}
	>;
	total: number;
};

export type RawInspectionRow = Partial<
	Record<InspectionColumnKey | "id", unknown>
> & {
	dataAkceptacjiNotyList?: unknown;
	dataZalecenList?: unknown;
	brakDataDoreczeniaPisma?: unknown;
	brakDataPismaZastrzezenia?: unknown;
	brakDataWyslaniaPismaZZastrzezeniami?: unknown;
	brakDataWplywuPisma?: unknown;
	brakDataPismaZOdpowiedzia?: unknown;
	brakDataWyslaniaPismaZOdpowiedzia?: unknown;
};

export type InspectionPeopleOption = {
	id: number;
	login: string;
	displayName: string;
	active: boolean;
	teamId: number | null;
	teamName: string | null;
	visibleOnList: boolean;
	canBeLeader: boolean;
	createdByOperator: boolean;
};

export type DictionarySelectOption = {
	value: string;
	label: string;
};

export function getBaseInspectionColumnKeys(
	viewId: InspectionViewId,
): InspectionColumnKey[] {
	if (viewId === "podstawowe") {
		return PODSTAWOWE_INSPECTION_COLUMN_KEYS;
	}

	if (viewId === "terminy") {
		return TERMINY_INSPECTION_COLUMN_KEYS;
	}

	return ALL_INSPECTION_COLUMN_KEYS;
}

export function toSafeString(value: unknown) {
	if (value === null || value === undefined) {
		return "";
	}

	return String(value);
}

function toSafeNumber(value: unknown, fallback: number) {
	if (typeof value === "number" && Number.isFinite(value)) {
		return value;
	}

	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : fallback;
}

function resolveRenumberedInspectionLp(rowIndex: number) {
	return rowIndex + 1;
}

function resolveInspectionCode(rawRow: RawInspectionRow) {
	const inspectionKod = toSafeString((rawRow as { inspectionKod?: unknown }).inspectionKod).trim();
	if (inspectionKod) {
		return inspectionKod;
	}

	const kodInspekcji = toSafeString((rawRow as { kodInspekcji?: unknown }).kodInspekcji).trim();
	if (kodInspekcji) {
		return kodInspekcji;
	}

	const lp = Number(rawRow.lp);
	if (Number.isFinite(lp) && lp > 0) {
		return String(lp);
	}

	const id = Number(rawRow.id);
	if (Number.isFinite(id) && id > 0) {
		return String(id);
	}

	return "-";
}

export function toDateOrSpecialValue(value: unknown, specialValue: string) {
	const normalized = toSafeString(value).trim();

	if (!normalized) {
		return "";
	}

	if (normalized.toLowerCase() === specialValue) {
		return specialValue;
	}

	return toDateInputValue(normalized);
}

export function normalizeInspectionRow(
	rawRow: RawInspectionRow,
	rowIndex: number,
): InspectionRow {
	const dataAkceptacjiNotyList = toDateList(rawRow.dataAkceptacjiNotyList);
	const dataZalecenList = toDateList(rawRow.dataZalecenList);

	return {
		id: toSafeString(rawRow.id) || `ins-row-${rowIndex + 1}`,
		lp: resolveRenumberedInspectionLp(rowIndex),
		kodInspekcji: resolveInspectionCode(rawRow),
		nazwaPodmiotu: toSafeString(rawRow.nazwaPodmiotu),
		typInspekcji: toSafeString(rawRow.typInspekcji),
		zakresInspekcji: toSafeString(rawRow.zakresInspekcji),
		szczegolyDotyczaceZakresu: toSafeString(rawRow.szczegolyDotyczaceZakresu),
		aspektKonsumencki: toSafeString(rawRow.aspektKonsumencki),
		poczatekInspekcji: toSafeString(rawRow.poczatekInspekcji),
		koniecInspekcji: toSafeString(rawRow.koniecInspekcji),
		osobaKierujaca: toSafeString(rawRow.osobaKierujaca),
		skladZespolu: toSafeString(rawRow.skladZespolu),
		rynek: toSafeString(rawRow.rynek),
		rodzajPodmiotu: toSafeString(rawRow.rodzajPodmiotu),
		dataProtokolu: toSafeString(rawRow.dataProtokolu),
		dataDoreczeniaProtokolu: toSafeString(rawRow.dataDoreczeniaProtokolu),
		dataAkceptacjiSprawozdania: toSafeString(rawRow.dataAkceptacjiSprawozdania),
		dataDoreczeniaPisma: toSafeString(rawRow.dataDoreczeniaPisma),
		dataPismaZastrzezenia: toSafeString(rawRow.dataPismaZastrzezenia),
		dataWyslaniaPismaZZastrzezeniami: toSafeString(
			rawRow.dataWyslaniaPismaZZastrzezeniami,
		),
		dataWplywuPisma: toSafeString(rawRow.dataWplywuPisma),
		dataPismaZOdpowiedzia: toSafeString(rawRow.dataPismaZOdpowiedzia),
		dataWyslaniaPismaZOdpowiedzia: toSafeString(
			rawRow.dataWyslaniaPismaZOdpowiedzia,
		),
		dataAkceptacjiNoty: dataAkceptacjiNotyList.length
			? dataAkceptacjiNotyList.join(", ")
			: toSafeString(rawRow.dataAkceptacjiNoty),
		dataZalecen: dataZalecenList.length
			? dataZalecenList.join(", ")
			: toSafeString(rawRow.dataZalecen),
		status: toSafeString(rawRow.status),
		komentarz: toSafeString(rawRow.komentarz),
	};
}

export function mapRowToAddForm(row: InspectionRow): AddInspectionForm {
	const normalizeOptionalSelectValue = (value: unknown) => {
		const normalized = toSafeString(value).trim();
		const lowered = normalized.toLowerCase();

		if (!normalized || lowered === "brak" || normalized === "-") {
			return "";
		}

		return normalized;
	};

	return {
		nazwaPodmiotu: toSafeString(row.nazwaPodmiotu),
		typInspekcji: toSafeString(row.typInspekcji),
		zakresInspekcji: toSafeString(row.zakresInspekcji),
		szczegolyDotyczaceZakresu: toSafeString(row.szczegolyDotyczaceZakresu),
		aspektKonsumencki: toSafeString(row.aspektKonsumencki),
		poczatekInspekcji: toSafeString(row.poczatekInspekcji),
		koniecInspekcji: toSafeString(row.koniecInspekcji),
		osobaKierujaca: toSafeString(row.osobaKierujaca),
		skladZespolu: toSafeString(row.skladZespolu),
		rynek: normalizeOptionalSelectValue(row.rynek),
		rodzajPodmiotu: normalizeOptionalSelectValue(row.rodzajPodmiotu),
		dataProtokolu: toDateInputValue(row.dataProtokolu),
		dataDoreczeniaProtokolu: toDateInputValue(row.dataDoreczeniaProtokolu),
		dataAkceptacjiSprawozdania: toDateInputValue(
			row.dataAkceptacjiSprawozdania,
		),
		dataDoreczeniaPisma: toDateInputValue(row.dataDoreczeniaPisma),
		dataPismaZastrzezenia: toDateInputValue(
			row.dataPismaZastrzezenia,
		),
		dataWyslaniaPismaZZastrzezeniami: toDateInputValue(
			row.dataWyslaniaPismaZZastrzezeniami,
		),
		dataWplywuPisma: toDateInputValue(row.dataWplywuPisma),
		dataPismaZOdpowiedzia: toDateInputValue(
			row.dataPismaZOdpowiedzia,
		),
		dataWyslaniaPismaZOdpowiedzia: toDateInputValue(
			row.dataWyslaniaPismaZOdpowiedzia,
		),
		dataAkceptacjiNoty: toDateOrSpecialValue(row.dataAkceptacjiNoty, "-"),
		dataZalecen: toSafeString(row.dataZalecen).trim(),
		status: toSafeString(row.status),
		komentarz: toSafeString(row.komentarz),
		brakDataDoreczeniaPisma: false,
		brakDataPismaZastrzezenia: false,
		brakDataWyslaniaPismaZZastrzezeniami: false,
		brakDataWplywuPisma: false,
		brakDataPismaZOdpowiedzia: false,
		brakDataWyslaniaPismaZOdpowiedzia: false,
	};
}

export function mapDictionaryEntriesToOptions(entries: DictionaryEntry[]) {
	const mappedOptions = entries
		.filter((entry) => entry.aktywny)
		.sort((left, right) => {
			const leftOrder = left.kolejnosc ?? Number.MAX_SAFE_INTEGER;
			const rightOrder = right.kolejnosc ?? Number.MAX_SAFE_INTEGER;
			if (leftOrder !== rightOrder) {
				return leftOrder - rightOrder;
			}

			return left.nazwaPozycji.localeCompare(right.nazwaPozycji, "pl", {
				sensitivity: "base",
			});
		})
		.map((entry) => entry.nazwaPozycji.trim())
		.filter(Boolean);

	return Array.from(new Set(mappedOptions));
}

export function mapDictionaryEntriesToSelectOptions(
	entries: DictionaryEntry[],
) {
	const mappedOptions = entries
		.filter((entry) => entry.aktywny)
		.sort((left, right) => {
			const leftOrder = left.kolejnosc ?? Number.MAX_SAFE_INTEGER;
			const rightOrder = right.kolejnosc ?? Number.MAX_SAFE_INTEGER;
			if (leftOrder !== rightOrder) {
				return leftOrder - rightOrder;
			}

			return left.nazwaPozycji.localeCompare(right.nazwaPozycji, "pl", {
				sensitivity: "base",
			});
		})
		.map((entry) => {
			const value = entry.nazwaPozycji.trim();
			const shortLabel = (entry.skrotPozycji ?? "").trim();

			return {
				value,
				label: shortLabel || value,
			};
		})
		.filter((option) => Boolean(option.value));

	const unique = new Map<string, DictionarySelectOption>();
	for (const option of mappedOptions) {
		if (!unique.has(option.value)) {
			unique.set(option.value, option);
		}
	}

	return Array.from(unique.values());
}

export function parseMultiValueField(value: string) {
	const normalized = value
		.split(";")
		.map((item) => item.trim())
		.filter(Boolean);

	return Array.from(new Set(normalized));
}

export function joinMultiValueField(values: string[]) {
	const normalized = values.map((item) => item.trim()).filter(Boolean);

	return Array.from(new Set(normalized)).join("; ");
}

export function getUserDisplayName(user: InspectionPeopleOption) {
	return user.displayName.trim() || user.login;
}

function stringifyApiErrorDetail(detail: unknown): string {
	if (!detail) {
		return "";
	}

	if (typeof detail === "string") {
		return detail;
	}

	if (Array.isArray(detail)) {
		const flattened = detail
			.map((item) => stringifyApiErrorDetail(item))
			.filter(Boolean)
			.join("; ");
		return flattened;
	}

	if (typeof detail === "object") {
		const maybeMessage =
			(detail as { msg?: unknown; message?: unknown }).msg ??
			(detail as { msg?: unknown; message?: unknown }).message;
		if (typeof maybeMessage === "string") {
			return maybeMessage;
		}

		try {
			return JSON.stringify(detail);
		} catch {
			return "";
		}
	}

	return String(detail);
}

export async function getInspectionApiErrorMessage(
	response: Response,
	fallbackMessage: string,
): Promise<string> {
	const statusMessageMap: Record<number, string> = {
		400: "Błędny payload biznesowy.",
		401: "Brak autoryzacji operatora (X-Operator-Login) lub operator nie istnieje.",
		403: "Brak uprawnień do tej operacji lub użytkownik jest nieaktywny.",
		404: "Nie znaleziono wskazanego zasobu.",
		409: "Konflikt danych biznesowych.",
		422: "Nieprawidłowy format danych (walidacja schematu JSON).",
	};

	const baseMessage =
		statusMessageMap[response.status] ??
		`${fallbackMessage} (${response.status}).`;

	const contentType = response.headers.get("content-type") ?? "";
	if (!contentType.includes("application/json")) {
		return baseMessage;
	}

	try {
		const payload = (await response.json()) as {
			detail?: unknown;
			message?: unknown;
		};
		const detail = payload.detail ?? payload.message;
		const detailMessage = stringifyApiErrorDetail(detail);
		if (
			response.status === 422 &&
			detailMessage
				.toLowerCase()
				.includes("listvisibility") &&
			(detailMessage.toLowerCase().includes("hidden") ||
				detailMessage.toLowerCase().includes("ukryty"))
		) {
			return "Nieprawidłowa widoczność członka zespołu: użytkownik ukryty nie może być dodany do składu zespołu.";
		}
		if (!detailMessage) {
			return baseMessage;
		}

		return `${baseMessage} ${detailMessage}`;
	} catch {
		return baseMessage;
	}
}
