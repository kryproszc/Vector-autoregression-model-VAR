import type { MenuSection, PanelMode } from "@/app/_components/home-tabs/types";

const REGISTRY_MENU_SECTIONS: MenuSection[] = [
	{
		id: "start",
		label: "Start",
		icon: "registers",
		items: [{ id: "start", label: "Start" }],
	},
	{
		id: "registers",
		label: "Rejestry",
		icon: "registers",
		items: [
			{ id: "inspections", label: "Inspekcje" },
			{ id: "recommendations", label: "Zalecenia" },
			{ id: "binding_decisions", label: "Decyzje zobowiązujące" },
			{ id: "sanction_requests", label: "Wnioski sankcyjne" },
		],
	},
	{
		id: "tools",
		label: "Raporty",
		icon: "tools",
		items: [
			{ id: "reports", label: "Wykonane inspekcje" },
			{ id: "report_protocol_time", label: "Czas protokołu" },
			{ id: "report_statement_time", label: "Czas sprawozdania" },
			{ id: "report_own_time", label: "Mój raport czasu" },
		],
	},
];

const MANAGEMENT_MENU_SECTIONS: MenuSection[] = [
	{
		id: "management",
		label: "Panel Zarządzania",
		icon: "tools",
		items: [
			{ id: "dictionaries", label: "Słowniki" },
			{ id: "teams", label: "Zespoły" },
			{ id: "users", label: "Użytkownicy" },
			{ id: "schedules", label: "Harmonogramy" },
			{ id: "logs", label: "Logi" },
		],
	},
];

export const PANEL_MODE_CONFIG: Record<
	PanelMode,
	{
		title: string;
		subtitle: string;
		sections: MenuSection[];
	}
> = {
	registry: {
		title: "Rejestr",
		subtitle: "Nawigacja modułów systemu",
		sections: REGISTRY_MENU_SECTIONS,
	},
	management: {
		title: "Panel Zarządzania",
		subtitle: "Wersja robocza dla kierownika i dyrektora",
		sections: MANAGEMENT_MENU_SECTIONS,
	},
};

export function getDefaultMenuItemId(mode: PanelMode) {
	return PANEL_MODE_CONFIG[mode].sections[0]?.items[0]?.id ?? "";
}
