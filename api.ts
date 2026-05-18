import type {
	InspectionStageSummaryRawGroup,
	InspectionStageSummaryRawSubgroup,
	ReportInspectionDetailedRawRow,
	ReportInspectionDetailedRow,
	ReportInspectionMatrixRawRow,
	ReportInspectionMatrixRow,
	ReportRecommendationDetailedRawRow,
	RecommendationStageSummaryRawGroup,
	ReportsRecommendationsDetailedResponse,
	ReportsRecommendationsStageSummaryResponse,
	ReportsInspectionsStageSummaryResponse,
	ReportsInspectionsTimeAnalyticsResponse,
	ReportsInspectionsDetailedResponse,
	ReportsInspectionsMatrixResponse,
	TimeReportTrendMode,
} from "@/features/reports/types";

type ApiResult<T> =
	| { ok: true; data: T }
	| { ok: false; error: string; status?: number };

function getErrorMessageByStatus(status: number) {
	if (status === 400) return "Błędne parametry raportu.";
	if (status === 401) return "Brak autoryzacji operatora.";
	if (status === 403) return "Brak uprawnień do raportów.";
	if (status === 404) return "Nie znaleziono endpointu raportów.";
	return `Błąd API (${status}).`;
}

async function readApiDetail(response: Response) {
	const contentType = response.headers.get("content-type") ?? "";
	if (!contentType.includes("application/json")) {
		return "";
	}

	try {
		const payload = (await response.json()) as {
			detail?: unknown;
			message?: unknown;
		};
		const detail = payload.detail ?? payload.message;
		return typeof detail === "string" ? detail : "";
	} catch {
		return "";
	}
}

function toCellValue(value: unknown) {
	if (typeof value === "string") {
		const trimmed = value.trim();
		return trimmed || "-";
	}

	if (typeof value === "number" && Number.isFinite(value)) {
		return String(value);
	}

	return "-";
}

function mapRawValues(rawValues: unknown) {
	if (!rawValues || typeof rawValues !== "object") {
		return {};
	}

	const entries = Object.entries(rawValues as Record<string, unknown>).map(
		([year, value]) => [year, toCellValue(value)] as const,
	);

	return Object.fromEntries(entries);
}

function mapRawRowToMatrixRow(
	rawRow: ReportInspectionMatrixRawRow,
): ReportInspectionMatrixRow {
	return {
		kodInspekcji: toCellValue(rawRow.kod_inspekcji),
		nazwaPodmiotu: toCellValue(rawRow.nazwa_podmiotu),
		wartosci: mapRawValues(rawRow.wartosci),
	};
}

function toNullableCellValue(value: unknown) {
	if (value === null || value === undefined) {
		return "-";
	}

	return toCellValue(value);
}

function toInspectionType(value: unknown): "K" | "W" | "-" {
	if (typeof value !== "string") {
		return "-";
	}

	const normalized = value.trim().toUpperCase();
	if (normalized === "K" || normalized === "W") {
		return normalized;
	}

	return "-";
}

function toDaysDifference(value: unknown) {
	if (value === null || value === undefined) {
		return "-";
	}

	if (typeof value === "number" && Number.isFinite(value)) {
		return String(value);
	}

	if (typeof value === "string") {
		const trimmed = value.trim();
		return trimmed || "-";
	}

	return "-";
}

function toNullableNumber(value: unknown) {
	if (typeof value === "number" && Number.isFinite(value)) {
		return value;
	}

	if (typeof value === "string") {
		const trimmed = value.trim();
		if (!trimmed) {
			return null;
		}

		const parsed = Number(trimmed);
		return Number.isFinite(parsed) ? parsed : null;
	}

	return null;
}

function toBoolean(value: unknown) {
	return value === true;
}

function toStageCode(value: unknown) {
	if (typeof value !== "string") {
		return "";
	}

	return value.trim().toLowerCase();
}

function toNonNegativeNumber(value: unknown) {
	const parsed = toNullableNumber(value);
	if (parsed === null) {
		return 0;
	}

	return parsed < 0 ? 0 : parsed;
}

function mapRawStageSubgroup(
	rawSubgroup: InspectionStageSummaryRawSubgroup,
) {
	return {
		stageSubgroupCode: toStageCode(
			rawSubgroup.stageSubgroupCode ?? rawSubgroup.stage_subgroup_code,
		),
		stageSubgroupLabel: toCellValue(
			rawSubgroup.stageSubgroupLabel ?? rawSubgroup.stage_subgroup_label,
		),
		stageSubgroupOrder: toNonNegativeNumber(
			rawSubgroup.stageSubgroupOrder ?? rawSubgroup.stage_subgroup_order,
		),
		count: toNonNegativeNumber(rawSubgroup.count),
		countTeam: toNonNegativeNumber(
			rawSubgroup.countTeam ?? rawSubgroup.count_team,
		),
		countManagerAdded: toNonNegativeNumber(
			rawSubgroup.countManagerAdded ?? rawSubgroup.count_manager_added,
		),
		countTeamAndManagerAdded: toNonNegativeNumber(
			rawSubgroup.countTeamAndManagerAdded ??
				rawSubgroup.count_team_and_manager_added,
		),
	};
}

function mapRawStageGroup(rawGroup: InspectionStageSummaryRawGroup) {
	const rawSubgroups = Array.isArray(rawGroup.subgroups)
		? rawGroup.subgroups
		: [];

	const subgroups = rawSubgroups
		.map((subgroup) => mapRawStageSubgroup((subgroup ?? {}) as InspectionStageSummaryRawSubgroup))
		.sort((left, right) => left.stageSubgroupOrder - right.stageSubgroupOrder);

	return {
		stageGroupCode: toStageCode(rawGroup.stageGroupCode ?? rawGroup.stage_group_code),
		stageGroupLabel: toCellValue(rawGroup.stageGroupLabel ?? rawGroup.stage_group_label),
		stageGroupOrder: toNonNegativeNumber(
			rawGroup.stageGroupOrder ?? rawGroup.stage_group_order,
		),
		count: toNonNegativeNumber(rawGroup.count),
		countTeam: toNonNegativeNumber(rawGroup.countTeam ?? rawGroup.count_team),
		countManagerAdded: toNonNegativeNumber(
			rawGroup.countManagerAdded ?? rawGroup.count_manager_added,
		),
		countTeamAndManagerAdded: toNonNegativeNumber(
			rawGroup.countTeamAndManagerAdded ?? rawGroup.count_team_and_manager_added,
		),
		subgroups,
	};
}

function mapRawRecommendationStageGroup(rawGroup: RecommendationStageSummaryRawGroup) {
	return {
		stageGroupCode: toStageCode(rawGroup.stageGroupCode ?? rawGroup.stage_group_code),
		stageGroupLabel: toCellValue(rawGroup.stageGroupLabel ?? rawGroup.stage_group_label),
		stageGroupOrder: toNonNegativeNumber(
			rawGroup.stageGroupOrder ?? rawGroup.stage_group_order,
		),
		count: toNonNegativeNumber(rawGroup.count),
		countTeam: toNonNegativeNumber(rawGroup.countTeam ?? rawGroup.count_team),
		countManagerAdded: toNonNegativeNumber(
			rawGroup.countManagerAdded ?? rawGroup.count_manager_added,
		),
		countTeamAndManagerAdded: toNonNegativeNumber(
			rawGroup.countTeamAndManagerAdded ?? rawGroup.count_team_and_manager_added,
		),
	};
}

function mapRawRowToDetailedRow(
	rawRow: ReportInspectionDetailedRawRow,
): ReportInspectionDetailedRow {
	const typInspekcji = toCellValue(rawRow.typ_inspekcji);
	const zakresInspekcji =
		toCellValue(rawRow.zakres_inspekcji) !== "-"
			? toCellValue(rawRow.zakres_inspekcji)
			: toCellValue(rawRow.typ_zakres_inspekcji);
	const inspektorKierujacy =
		toCellValue(rawRow.inspektor_kierujacy) !== "-"
			? toCellValue(rawRow.inspektor_kierujacy)
			: toCellValue(rawRow.osoba_kierujaca);
	const status =
		toCellValue(rawRow.status) !== "-"
			? toCellValue(rawRow.status)
			: toCellValue(rawRow.status_inspekcji);

	return {
		kodInspekcji: toCellValue(rawRow.kod_inspekcji),
		nazwaPodmiotu: toCellValue(rawRow.nazwa_podmiotu),
		rodzajPodmiotu: toCellValue(rawRow.rodzaj_podmiotu),
		typInspekcji: typInspekcji,
		inspekcja: toInspectionType(rawRow.inspekcja),
		zakresInspekcji,
		inspektorKierujacy,
		kontrola: zakresInspekcji,
		rokPoczatku: toCellValue(rawRow.rok_poczatku),
		poczatekInspekcji: toCellValue(rawRow.poczatek_inspekcji),
		koniecInspekcji: toCellValue(rawRow.koniec_inspekcji),
		osobaKierujaca: inspektorKierujacy,
		zespol: toCellValue(rawRow.zespol_osoby_kierujacej_kod),
		data: toNullableCellValue(rawRow.data_protokolu_sprawozdania),
		czas: toDaysDifference(rawRow.roznica_dni_miedzy_data_protokolu_a_koncem),
		liczbaDniOdKoncaInspekcjiDoDzis: toNullableNumber(
			rawRow.liczba_dni_od_konca_inspekcji_do_dzis,
		),
		wartoscLiczbowaPrzedzialu: toNullableNumber(rawRow.wartosc_liczbowa_przedzialu),
		wartoscLiczbowaPrzedzialuAlt: toNullableNumber(
			rawRow.wartosc_liczbowa_przedzialu_alt,
		),
		statusInspekcjiId: toNullableNumber(rawRow.status_inspekcji_id),
		statusInspekcji: status,
		isLeaderCurrentUser: toBoolean(rawRow.is_leader_current_user),
		isLeaderInManagerTeam: toBoolean(rawRow.is_leader_in_manager_team),
		isMemberCurrentUser: toBoolean(rawRow.is_member_current_user),
		isMemberInManagerTeam: toBoolean(rawRow.is_member_in_manager_team),
		stageGroupCode: toStageCode(rawRow.stage_group_code),
		stageSubgroupCode: toStageCode(rawRow.stage_subgroup_code),
	};
}

function mapRawRowToRecommendationDetailedRow(
	rawRow: ReportRecommendationDetailedRawRow,
) {
	return {
		status: toCellValue(rawRow.status),
		recommendationId: toCellValue(
			rawRow.kod_zalecenia ?? rawRow.kodZalecenia ?? rawRow.recommendation_id ?? rawRow.recommendationId,
		),
		inspectionId: toCellValue(
			rawRow.kod_inspekcji ?? rawRow.kodInspekcji ?? rawRow.inspection_id ?? rawRow.inspectionId,
		),
		nazwaPodmiotu: toCellValue(rawRow.nazwa_podmiotu ?? rawRow.nazwaPodmiotu),
		dataZalecen: toCellValue(rawRow.data_zalecen ?? rawRow.dataZalecen),
		terminZalecen: toCellValue(rawRow.termin_zalecen ?? rawRow.terminZalecen),
		terminWykonaniaZalecen: toCellValue(
			rawRow.termin_wykonania_zalecen ?? rawRow.terminWykonaniaZalecen,
		),
		liczbaZalecen: toCellValue(rawRow.liczba_zalecen ?? rawRow.liczbaZalecen),
	};
}

export async function fetchInspectionsReportMatrix(
	operatorLogin: string,
): Promise<ApiResult<ReportsInspectionsMatrixResponse>> {
	const response = await fetch("/api/reports/inspections-matrix", {
		method: "GET",
		headers: {
			"Content-Type": "application/json",
			"X-Operator-Login": operatorLogin,
		},
		cache: "no-store",
	});

	if (!response.ok) {
		const detail = await readApiDetail(response);
		const base = getErrorMessageByStatus(response.status);
		return {
			ok: false,
			error: detail ? `${base} ${detail}` : base,
			status: response.status,
		};
	}

	const payload = (await response.json()) as Partial<{
		lata: unknown;
		rows: unknown;
	}>;

	const years = Array.isArray(payload.lata)
		? payload.lata
				.map((year) => String(year ?? "").trim())
				.filter(Boolean)
		: [];
	const rawRows = Array.isArray(payload.rows) ? payload.rows : [];

	return {
		ok: true,
		data: {
			lata: years,
			rows: rawRows.map((row) =>
				mapRawRowToMatrixRow((row ?? {}) as ReportInspectionMatrixRawRow),
			),
		},
	};
}

export async function fetchInspectionsDetailedReport(
	operatorLogin: string,
): Promise<ApiResult<ReportsInspectionsDetailedResponse>> {
	const response = await fetch("/api/reports/inspections-detailed", {
		method: "GET",
		headers: {
			"Content-Type": "application/json",
			"X-Operator-Login": operatorLogin,
		},
		cache: "no-store",
	});

	if (!response.ok) {
		const detail = await readApiDetail(response);
		const base = getErrorMessageByStatus(response.status);
		return {
			ok: false,
			error: detail ? `${base} ${detail}` : base,
			status: response.status,
		};
	}

	const payload = (await response.json()) as Partial<{
		rows: unknown;
	}>;

	const rawRows = Array.isArray(payload.rows) ? payload.rows : [];

	return {
		ok: true,
		data: {
			rows: rawRows.map((row) =>
				mapRawRowToDetailedRow((row ?? {}) as ReportInspectionDetailedRawRow),
			),
		},
	};
}

export async function fetchInspectionsStageSummary(
	operatorLogin: string,
	params?: {
		managerUserId?: string | number | null;
	},
): Promise<ApiResult<ReportsInspectionsStageSummaryResponse>> {
	const searchParams = new URLSearchParams();
	if (params?.managerUserId !== null && params?.managerUserId !== undefined) {
		const normalizedManagerUserId = String(params.managerUserId).trim();
		if (normalizedManagerUserId) {
			searchParams.set("managerUserId", normalizedManagerUserId);
		}
	}

	const requestUrl = searchParams.toString()
		? `/api/reports/inspections-stage-summary?${searchParams.toString()}`
		: "/api/reports/inspections-stage-summary";

	const response = await fetch(requestUrl, {
		method: "GET",
		headers: {
			"Content-Type": "application/json",
			"X-Operator-Login": operatorLogin,
		},
		cache: "no-store",
	});

	if (!response.ok) {
		const detail = await readApiDetail(response);
		const base = getErrorMessageByStatus(response.status);
		return {
			ok: false,
			error: detail ? `${base} ${detail}` : base,
			status: response.status,
		};
	}

	const payload = (await response.json()) as Partial<{
		generatedAt: unknown;
		generated_at: unknown;
		stageDictionaryVersion: unknown;
		stage_dictionary_version: unknown;
		totalInspections: unknown;
		total_inspections: unknown;
		qualityErrorCount: unknown;
		quality_error_count: unknown;
		groups: unknown;
	}>;

	const rawGroups = Array.isArray(payload.groups) ? payload.groups : [];
	const groups = rawGroups
		.map((group) => mapRawStageGroup((group ?? {}) as InspectionStageSummaryRawGroup))
		.sort((left, right) => left.stageGroupOrder - right.stageGroupOrder);

	return {
		ok: true,
		data: {
			generatedAt: String(payload.generatedAt ?? payload.generated_at ?? "").trim(),
			stageDictionaryVersion: String(
				payload.stageDictionaryVersion ?? payload.stage_dictionary_version ?? "",
			).trim(),
			totalInspections: toNonNegativeNumber(
				payload.totalInspections ?? payload.total_inspections,
			),
			qualityErrorCount: toNonNegativeNumber(
				payload.qualityErrorCount ?? payload.quality_error_count,
			),
			groups,
		},
	};
}

export async function fetchInspectionsTimeAnalytics(
	operatorLogin: string,
	params: {
		inspectionType: "K" | "W";
		trendMode: TimeReportTrendMode;
		teams: string[];
		years: string[];
	},
): Promise<ApiResult<ReportsInspectionsTimeAnalyticsResponse>> {
	const searchParams = new URLSearchParams({
		inspectionType: params.inspectionType,
		trendMode: params.trendMode,
	});

	if (params.teams.length > 0) {
		searchParams.set("teams", params.teams.join(","));
	}

	if (params.years.length > 0) {
		searchParams.set("years", params.years.join(","));
	}

	const response = await fetch(`/api/reports/inspections-time-analytics?${searchParams.toString()}`, {
		method: "GET",
		headers: {
			"Content-Type": "application/json",
			"X-Operator-Login": operatorLogin,
		},
		cache: "no-store",
	});

	if (!response.ok) {
		const detail = await readApiDetail(response);
		const base = getErrorMessageByStatus(response.status);
		return {
			ok: false,
			error: detail ? `${base} ${detail}` : base,
			status: response.status,
		};
	}

	const toNumber = (value: unknown, fallback = 0) => {
		if (typeof value === "number" && Number.isFinite(value)) {
			return value;
		}

		if (typeof value === "string") {
			const parsed = Number(value.replace(",", ".").trim());
			if (Number.isFinite(parsed)) {
				return parsed;
			}
		}

		return fallback;
	};

	const toStringArray = (value: unknown) =>
		Array.isArray(value)
			? value.map((item) => String(item ?? "").trim()).filter(Boolean)
			: [];

	const payload = (await response.json()) as Record<string, unknown>;
	const selectedMetricLabelFromBackend = String(
		payload.selectedMetricLabel ?? "",
	).trim();
	const selectedMetric =
		payload.selectedMetric === "average" || payload.selectedMetric === "median"
			? payload.selectedMetric
			: params.trendMode;
	const selectedMetricLabel =
		selectedMetricLabelFromBackend ||
		(params.trendMode === "average" ? "Średni czas" : "Mediana czasu");

	const alertStatusCounts = Array.isArray(payload.alertStatusCounts)
		? payload.alertStatusCounts
				.map((item) => {
					if (!item || typeof item !== "object") {
						return null;
					}

					const source = item as Record<string, unknown>;
					return {
						statusInspekcjiId: toNullableNumber(source.statusInspekcjiId),
						statusInspekcji: String(source.statusInspekcji ?? "").trim(),
						count: toNumber(source.count, 0),
					};
				})
				.filter(
					(
						item,
					): item is ReportsInspectionsTimeAnalyticsResponse["alertStatusCounts"][number] =>
						item !== null,
				)
		: [];

	const summaryPivotYears = toStringArray(payload.summaryPivotYears);

	const summaryPivotRows = Array.isArray(payload.summaryPivotRows)
		? payload.summaryPivotRows
				.map((row) => {
					if (!row || typeof row !== "object") {
						return null;
					}

					const source = row as Record<string, unknown>;
					const rawValues =
						source.values && typeof source.values === "object"
							? (source.values as Record<string, unknown>)
							: {};
					const values: Record<string, string | number | null> = {};
					for (const [year, value] of Object.entries(rawValues)) {
						if (value === null || value === undefined) {
							values[year] = null;
							continue;
						}

						if (typeof value === "number" && Number.isFinite(value)) {
							values[year] = value;
							continue;
						}

						if (typeof value === "string") {
							values[year] = value;
							continue;
						}

						values[year] = String(value);
					}

					return {
						zespol: String(source.zespol ?? "-"),
						values,
					};
				})
				.filter(
					(row): row is ReportsInspectionsTimeAnalyticsResponse["summaryPivotRows"][number] =>
						row !== null,
				)
		: [];

	const yearCountColumns = toStringArray(payload.yearCountColumns);

	const yearCountRows = Array.isArray(payload.yearCountRows)
		? payload.yearCountRows
				.map((row) => {
					if (!row || typeof row !== "object") {
						return null;
					}

					const source = row as Record<string, unknown>;
					const rawValues =
						source.values && typeof source.values === "object"
							? (source.values as Record<string, unknown>)
							: {};
					const values: Record<string, string | number | null> = {};
					for (const [year, value] of Object.entries(rawValues)) {
						if (value === null || value === undefined) {
							values[year] = null;
							continue;
						}

						if (typeof value === "number" && Number.isFinite(value)) {
							values[year] = value;
							continue;
						}

						if (typeof value === "string") {
							values[year] = value;
							continue;
						}

						values[year] = String(value);
					}

					return {
						label: String(source.label ?? "-"),
						values,
					};
				})
				.filter(
					(
						row,
					): row is ReportsInspectionsTimeAnalyticsResponse["yearCountRows"][number] =>
						row !== null,
				)
		: [];

	const yearCountByTeamColumns = toStringArray(payload.yearCountByTeamColumns);

	const yearCountByTeamRows = Array.isArray(payload.yearCountByTeamRows)
		? payload.yearCountByTeamRows
				.map((row) => {
					if (!row || typeof row !== "object") {
						return null;
					}

					const source = row as Record<string, unknown>;
					const rawValues =
						source.values && typeof source.values === "object"
							? (source.values as Record<string, unknown>)
							: {};
					const values: Record<string, string | number | null> = {};
					for (const [year, value] of Object.entries(rawValues)) {
						if (value === null || value === undefined) {
							values[year] = null;
							continue;
						}

						if (typeof value === "number" && Number.isFinite(value)) {
							values[year] = value;
							continue;
						}

						if (typeof value === "string") {
							values[year] = value;
							continue;
						}

						values[year] = String(value);
					}

					const inspekcja = source.inspekcja === "W" ? "W" : "K";
					const zespol = String(source.zespol ?? "-");
					const defaultLabel = `${zespol} - ${inspekcja === "K" ? "Kontrola" : "Wizyta nadzorcza"}`;

					return {
						label: String(source.label ?? defaultLabel),
						zespol,
						inspekcja,
						values,
					};
				})
				.filter(
					(
						row,
					): row is ReportsInspectionsTimeAnalyticsResponse["yearCountByTeamRows"][number] =>
						row !== null,
				)
		: [];

	const overallColumns = Array.isArray(payload.overallColumns)
		? payload.overallColumns
				.map((column) => {
					if (!column || typeof column !== "object") {
						return null;
					}

					const source = column as Record<string, unknown>;
					const key = String(source.key ?? "").trim();
					const label = String(source.label ?? "").trim();
					if (!key || !label) {
						return null;
					}

					return { key, label };
				})
				.filter(
					(
						column,
					): column is ReportsInspectionsTimeAnalyticsResponse["overallColumns"][number] =>
						column !== null,
				)
		: [];

	const overallRows = Array.isArray(payload.overallRows)
		? payload.overallRows
				.map((row) => {
					if (!row || typeof row !== "object") {
						return null;
					}

					const source = row as Record<string, unknown>;
					const normalized: Record<string, string | number | null> = {};
					for (const [key, value] of Object.entries(source)) {
						if (value === null || value === undefined) {
							normalized[key] = null;
							continue;
						}

						if (typeof value === "number" && Number.isFinite(value)) {
							normalized[key] = value;
							continue;
						}

						if (typeof value === "string") {
							normalized[key] = value;
							continue;
						}

						normalized[key] = String(value);
					}

					return normalized;
				})
				.filter(
					(row): row is ReportsInspectionsTimeAnalyticsResponse["overallRows"][number] =>
						row !== null,
				)
		: [];

	const trendRows = Array.isArray(payload.trendRows)
		? payload.trendRows
				.map((row) => {
					if (!row || typeof row !== "object") {
						return null;
					}

					const source = row as Record<string, unknown>;
					return {
						year: toNumber(source.year, 0),
						trend: toNumber(source.trend, 0),
						average: toNumber(source.average, 0),
						median: toNumber(source.median, 0),
						min: toNumber(source.min, 0),
						max: toNumber(source.max, 0),
						count: toNumber(source.count, 0),
					};
				})
				.filter(
					(row): row is ReportsInspectionsTimeAnalyticsResponse["trendRows"][number] =>
						row !== null,
				)
		: [];

	const scatterRows = Array.isArray(payload.scatterRows)
		? payload.scatterRows
				.map((row) => {
					if (!row || typeof row !== "object") {
						return null;
					}

					const source = row as Record<string, unknown>;
					return {
						year: toNumber(source.year, 0),
						time: toNumber(source.time, 0),
						nazwaPodmiotu: String(source.nazwaPodmiotu ?? ""),
						kontrola: String(source.kontrola ?? ""),
						osobaKierujaca: String(source.osobaKierujaca ?? ""),
						zespol: String(source.zespol ?? ""),
					};
				})
				.filter(
					(row): row is ReportsInspectionsTimeAnalyticsResponse["scatterRows"][number] =>
						row !== null,
				)
		: [];

	const detailRows = Array.isArray(payload.detailRows)
		? payload.detailRows
				.map((row) => {
					if (!row || typeof row !== "object") {
						return null;
					}

					const source = row as Record<string, unknown>;
					return {
						kodInspekcji: String(source.kodInspekcji ?? "-"),
						nazwaPodmiotu: String(source.nazwaPodmiotu ?? "-"),
						inspekcja:
							source.inspekcja === "K" || source.inspekcja === "W"
								? source.inspekcja
								: "-",
						kontrola: String(source.kontrola ?? "-"),
						rokPoczatku: String(source.rokPoczatku ?? "-"),
						poczatekInspekcji: String(source.poczatekInspekcji ?? "-"),
						koniecInspekcji: String(source.koniecInspekcji ?? "-"),
						osobaKierujaca: String(source.osobaKierujaca ?? "-"),
						zespol: String(source.zespol ?? "-"),
						data: String(source.data ?? "-"),
						czas: String(source.czas ?? "-"),
						statusInspekcjiId: toNullableNumber(source.statusInspekcjiId),
						statusInspekcji: String(source.statusInspekcji ?? "-").trim() || "-",
					};
				})
				.filter(
					(row): row is ReportsInspectionsTimeAnalyticsResponse["detailRows"][number] =>
						row !== null,
				)
		: [];

	return {
		ok: true,
		data: {
			inspectionType:
				payload.inspectionType === "W" ? "W" : "K",
			trendMode:
				payload.trendMode === "median" || payload.trendMode === "average"
					? payload.trendMode
					: params.trendMode,
			selectedMetric,
			selectedMetricLabel,
			baseCount: toNumber(payload.baseCount, 0),
			filteredCount: toNumber(payload.filteredCount, 0),
			alertStatusCounts,
			alertPiszemyProtokolCount: toNumber(payload.alertPiszemyProtokolCount, 0),
			teamOptions: toStringArray(payload.teamOptions),
			yearOptions: toStringArray(payload.yearOptions),
			detailRows,
			scatterRows,
			trendRows,
			summaryPivotYears,
			summaryPivotRows,
			yearCountColumns,
			yearCountRows,
			yearCountByTeamColumns,
			yearCountByTeamRows,
			overallColumns,
			overallRows,
		},
	};
}

export async function fetchRecommendationsStageSummary(
	operatorLogin: string,
	params?: {
		managerUserId?: string | number | null;
	},
): Promise<ApiResult<ReportsRecommendationsStageSummaryResponse>> {
	const searchParams = new URLSearchParams();
	if (params?.managerUserId !== null && params?.managerUserId !== undefined) {
		const normalizedManagerUserId = String(params.managerUserId).trim();
		if (normalizedManagerUserId) {
			searchParams.set("managerUserId", normalizedManagerUserId);
		}
	}

	const requestUrl = searchParams.toString()
		? `/api/reports/recommendations-stage-summary?${searchParams.toString()}`
		: "/api/reports/recommendations-stage-summary";

	const response = await fetch(requestUrl, {
		method: "GET",
		headers: {
			"Content-Type": "application/json",
			"X-Operator-Login": operatorLogin,
		},
		cache: "no-store",
	});

	if (!response.ok) {
		const detail = await readApiDetail(response);
		const base = getErrorMessageByStatus(response.status);
		return {
			ok: false,
			error: detail ? `${base} ${detail}` : base,
			status: response.status,
		};
	}

	const payload = (await response.json()) as Partial<{
		generatedAt: unknown;
		generated_at: unknown;
		stageDictionaryVersion: unknown;
		stage_dictionary_version: unknown;
		totalRecommendations: unknown;
		total_recommendations: unknown;
		qualityErrorCount: unknown;
		quality_error_count: unknown;
		groups: unknown;
	}>;

	const rawGroups = Array.isArray(payload.groups) ? payload.groups : [];
	const groups = rawGroups
		.map((group) =>
			mapRawRecommendationStageGroup((group ?? {}) as RecommendationStageSummaryRawGroup),
		)
		.sort((left, right) => left.stageGroupOrder - right.stageGroupOrder);

	return {
		ok: true,
		data: {
			generatedAt: String(payload.generatedAt ?? payload.generated_at ?? "").trim(),
			stageDictionaryVersion: String(
				payload.stageDictionaryVersion ?? payload.stage_dictionary_version ?? "",
			).trim(),
			totalRecommendations: toNonNegativeNumber(
				payload.totalRecommendations ?? payload.total_recommendations,
			),
			qualityErrorCount: toNonNegativeNumber(
				payload.qualityErrorCount ?? payload.quality_error_count,
			),
			groups,
		},
	};
}

export async function fetchRecommendationsDetailedReport(
	operatorLogin: string,
	params?: {
		managerUserId?: string | number | null;
	},
): Promise<ApiResult<ReportsRecommendationsDetailedResponse>> {
	const searchParams = new URLSearchParams();
	if (params?.managerUserId !== null && params?.managerUserId !== undefined) {
		const normalizedManagerUserId = String(params.managerUserId).trim();
		if (normalizedManagerUserId) {
			searchParams.set("managerUserId", normalizedManagerUserId);
		}
	}

	const requestUrl = searchParams.toString()
		? `/api/reports/recommendations-detailed?${searchParams.toString()}`
		: "/api/reports/recommendations-detailed";

	const response = await fetch(requestUrl, {
		method: "GET",
		headers: {
			"Content-Type": "application/json",
			"X-Operator-Login": operatorLogin,
		},
		cache: "no-store",
	});

	if (!response.ok) {
		const detail = await readApiDetail(response);
		const base = getErrorMessageByStatus(response.status);
		return {
			ok: false,
			error: detail ? `${base} ${detail}` : base,
			status: response.status,
		};
	}

	const payload = (await response.json()) as Partial<{
		rows: unknown;
	}>;

	const rawRows = Array.isArray(payload.rows) ? payload.rows : [];

	return {
		ok: true,
		data: {
			rows: rawRows.map((row) =>
				mapRawRowToRecommendationDetailedRow((row ?? {}) as ReportRecommendationDetailedRawRow),
			),
		},
	};
}
