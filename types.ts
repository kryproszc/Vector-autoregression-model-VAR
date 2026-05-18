export type ReportInspectionMatrixRawRow = {
	kod_inspekcji?: unknown;
	nazwa_podmiotu?: unknown;
	wartosci?: unknown;
};

export type ReportInspectionMatrixRow = {
	kodInspekcji: string;
	nazwaPodmiotu: string;
	wartosci: Record<string, string>;
};

export type ReportsInspectionsMatrixResponse = {
	lata: string[];
	rows: ReportInspectionMatrixRow[];
};

export type ReportInspectionDetailedRawRow = {
	kod_inspekcji?: unknown;
	nazwa_podmiotu?: unknown;
	rodzaj_podmiotu?: unknown;
	typ_inspekcji?: unknown;
	inspekcja?: unknown;
	zakres_inspekcji?: unknown;
	typ_zakres_inspekcji?: unknown;
	rok_poczatku?: unknown;
	poczatek_inspekcji?: unknown;
	koniec_inspekcji?: unknown;
	inspektor_kierujacy?: unknown;
	osoba_kierujaca?: unknown;
	zespol_osoby_kierujacej_kod?: unknown;
	status?: unknown;
	data_protokolu_sprawozdania?: unknown;
	roznica_dni_miedzy_data_protokolu_a_koncem?: unknown;
	liczba_dni_od_konca_inspekcji_do_dzis?: unknown;
	wartosc_liczbowa_przedzialu?: unknown;
	wartosc_liczbowa_przedzialu_alt?: unknown;
	status_inspekcji_id?: unknown;
	status_inspekcji?: unknown;
	is_leader_current_user?: unknown;
	is_leader_in_manager_team?: unknown;
	is_member_current_user?: unknown;
	is_member_in_manager_team?: unknown;
	stage_group_code?: unknown;
	stage_subgroup_code?: unknown;
};

export type ReportInspectionDetailedRow = {
	kodInspekcji: string;
	nazwaPodmiotu: string;
	rodzajPodmiotu: string;
	typInspekcji: string;
	inspekcja: "K" | "W" | "-";
	zakresInspekcji: string;
	inspektorKierujacy: string;
	kontrola: string;
	rokPoczatku: string;
	poczatekInspekcji: string;
	koniecInspekcji: string;
	osobaKierujaca: string;
	zespol: string;
	data: string;
	czas: string;
	liczbaDniOdKoncaInspekcjiDoDzis: number | null;
	wartoscLiczbowaPrzedzialu: number | null;
	wartoscLiczbowaPrzedzialuAlt: number | null;
	statusInspekcjiId: number | null;
	statusInspekcji: string;
	isLeaderCurrentUser: boolean;
	isLeaderInManagerTeam: boolean;
	isMemberCurrentUser: boolean;
	isMemberInManagerTeam: boolean;
	stageGroupCode: string;
	stageSubgroupCode: string;
};

export type ReportsInspectionsDetailedResponse = {
	rows: ReportInspectionDetailedRow[];
};

export type InspectionStageSummaryRawSubgroup = {
	stageSubgroupCode?: unknown;
	stage_subgroup_code?: unknown;
	stageSubgroupLabel?: unknown;
	stage_subgroup_label?: unknown;
	stageSubgroupOrder?: unknown;
	stage_subgroup_order?: unknown;
	count?: unknown;
	countTeam?: unknown;
	count_team?: unknown;
	countManagerAdded?: unknown;
	count_manager_added?: unknown;
	countTeamAndManagerAdded?: unknown;
	count_team_and_manager_added?: unknown;
};

export type InspectionStageSummaryRawGroup = {
	stageGroupCode?: unknown;
	stage_group_code?: unknown;
	stageGroupLabel?: unknown;
	stage_group_label?: unknown;
	stageGroupOrder?: unknown;
	stage_group_order?: unknown;
	count?: unknown;
	countTeam?: unknown;
	count_team?: unknown;
	countManagerAdded?: unknown;
	count_manager_added?: unknown;
	countTeamAndManagerAdded?: unknown;
	count_team_and_manager_added?: unknown;
	subgroups?: unknown;
};

export type InspectionStageSummarySubgroup = {
	stageSubgroupCode: string;
	stageSubgroupLabel: string;
	stageSubgroupOrder: number;
	count: number;
	countTeam: number;
	countManagerAdded: number;
	countTeamAndManagerAdded: number;
};

export type InspectionStageSummaryGroup = {
	stageGroupCode: string;
	stageGroupLabel: string;
	stageGroupOrder: number;
	count: number;
	countTeam: number;
	countManagerAdded: number;
	countTeamAndManagerAdded: number;
	subgroups: InspectionStageSummarySubgroup[];
};

export type ReportsInspectionsStageSummaryResponse = {
	generatedAt: string;
	stageDictionaryVersion: string;
	totalInspections: number;
	qualityErrorCount: number;
	groups: InspectionStageSummaryGroup[];
};

export type RecommendationStageSummaryRawGroup = {
	stageGroupCode?: unknown;
	stage_group_code?: unknown;
	stageGroupLabel?: unknown;
	stage_group_label?: unknown;
	stageGroupShortLabel?: unknown;
	stage_group_short_label?: unknown;
	stageGroupOrder?: unknown;
	stage_group_order?: unknown;
	count?: unknown;
	countTeam?: unknown;
	count_team?: unknown;
	countManagerAdded?: unknown;
	count_manager_added?: unknown;
	countTeamAndManagerAdded?: unknown;
	count_team_and_manager_added?: unknown;
};

export type RecommendationStageSummaryGroup = {
	stageGroupCode: string;
	stageGroupLabel: string;
	stageGroupShortLabel: string;
	stageGroupOrder: number;
	count: number;
	countTeam: number;
	countManagerAdded: number;
	countTeamAndManagerAdded: number;
};

export type ReportsRecommendationsStageSummaryResponse = {
	generatedAt: string;
	stageDictionaryVersion: string;
	totalRecommendations: number;
	qualityErrorCount: number;
	groups: RecommendationStageSummaryGroup[];
};

export type ReportRecommendationDetailedRawRow = {
	status?: unknown;
	status_skrot?: unknown;
	statusSkrot?: unknown;
	kod_zalecenia?: unknown;
	kodZalecenia?: unknown;
	recommendation_id?: unknown;
	recommendationId?: unknown;
	kod_inspekcji?: unknown;
	kodInspekcji?: unknown;
	inspection_id?: unknown;
	inspectionId?: unknown;
	nazwa_podmiotu?: unknown;
	nazwaPodmiotu?: unknown;
	data_zalecen?: unknown;
	dataZalecen?: unknown;
	termin_zalecen?: unknown;
	terminZalecen?: unknown;
	termin_wykonania_zalecen?: unknown;
	terminWykonaniaZalecen?: unknown;
	liczba_zalecen?: unknown;
	liczbaZalecen?: unknown;
};

export type ReportRecommendationDetailedRow = {
	status: string;
	statusSkrot: string;
	recommendationId: string;
	inspectionId: string;
	nazwaPodmiotu: string;
	dataZalecen: string;
	terminZalecen: string;
	terminWykonaniaZalecen: string;
	liczbaZalecen: string;
};

export type ReportsRecommendationsDetailedResponse = {
	rows: ReportRecommendationDetailedRow[];
};

export type TimeReportTrendMode = "average" | "median";

export type TimeReportAlertStatusCount = {
	statusInspekcjiId: number | null;
	statusInspekcji: string;
	count: number;
};

export type TimeReportSummaryPivotRow = {
	zespol: string;
	values: Record<string, string | number | null>;
};

export type TimeReportYearCountRow = {
	label: string;
	values: Record<string, string | number | null>;
};

export type TimeReportYearCountByTeamRow = {
	label: string;
	zespol: string;
	inspekcja: "K" | "W";
	values: Record<string, string | number | null>;
};

export type TimeReportOverallRow = {
	[key: string]: string | number | null;
};

export type TimeReportOverallColumn = {
	key: string;
	label: string;
};

export type TimeReportTrendRow = {
	year: number;
	trend: number;
	average: number;
	median: number;
	min: number;
	max: number;
	count: number;
};

export type TimeReportScatterRow = {
	year: number;
	time: number;
	nazwaPodmiotu: string;
	kontrola: string;
	osobaKierujaca: string;
	zespol: string;
};

export type ReportsInspectionsTimeAnalyticsResponse = {
	inspectionType: "K" | "W";
	trendMode: TimeReportTrendMode;
	selectedMetric: TimeReportTrendMode;
	selectedMetricLabel: string;
	baseCount: number;
	filteredCount: number;
	alertStatusCounts: TimeReportAlertStatusCount[];
	alertPiszemyProtokolCount: number;
	teamOptions: string[];
	yearOptions: string[];
	detailRows: ReportInspectionDetailedRow[];
	scatterRows: TimeReportScatterRow[];
	trendRows: TimeReportTrendRow[];
	summaryPivotYears: string[];
	summaryPivotRows: TimeReportSummaryPivotRow[];
	yearCountColumns: string[];
	yearCountRows: TimeReportYearCountRow[];
	yearCountByTeamColumns: string[];
	yearCountByTeamRows: TimeReportYearCountByTeamRow[];
	overallColumns: TimeReportOverallColumn[];
	overallRows: TimeReportOverallRow[];
};
