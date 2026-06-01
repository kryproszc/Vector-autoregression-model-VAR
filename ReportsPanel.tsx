import { AdminUsersPanel } from "@/features/admin-users/components/AdminUsersPanel";
import { DictionariesSheetPanel } from "@/features/dictionaries/components/DictionariesSheetPanel";
import { ExternalUsersPanel } from "@/features/external-users/components/ExternalUsersPanel";
import { UsersPanel } from "@/features/users/components/UsersPanel";
import { InspectionsPanel } from "@/features/inspections/components/InspectionsPanel";
import { ObligatingDecisionsPanel } from "@/features/obligating-decisions/components/ObligatingDecisionsPanel";
import { RecommendationsPanel } from "@/features/recommendations/components/RecommendationsPanel";
import {
	ProtocolTimePanel,
	StatementTimePanel,
} from "@/features/reports/components/ProtocolTimePanel";
import { OwnTimeReportPanel } from "@/features/reports/components/OwnTimeReportPanel";
import { ReportsPanel } from "@/features/reports/components/ReportsPanel";
import { SanctionRequestsPanel } from "@/features/sanction-requests/components/SanctionRequestsPanel";
import { WelcomeStartPanel } from "@/features/welcome/components/WelcomeStartPanel";
import { PlaceholderPanel } from "@/shared/components/PlaceholderPanel";
import { SchedulesPanel } from "@/features/schedules/components/SchedulesPanel";
import { LogsPanel } from "@/features/logs/components/LogsPanel";

import type { AuthRole, PanelMode } from "@/app/_components/home-tabs/types";

type RenderTabContentParams = {
	panelMode: PanelMode;
	selectedItemId: string;
	operatorLogin: string;
	authRole: AuthRole;
	isObserver: boolean;
	canEditDictionaries: boolean;
};

type TabRenderer = (params: RenderTabContentParams) => React.ReactNode;

const TAB_CONTENT_RENDERERS: Partial<
	Record<PanelMode, Record<string, TabRenderer>>
> = {
	registry: {
		start: ({ operatorLogin }) => (
			<WelcomeStartPanel operatorLogin={operatorLogin} />
		),
		dictionaries: ({ operatorLogin, canEditDictionaries, authRole }) => (
			<DictionariesSheetPanel
				key="dictionaries-readonly"
				operatorLogin={operatorLogin}
				authRole={authRole}
				title="Słowniki"
				subtitle=""
				canEdit={canEditDictionaries}
			/>
		),
		inspections: ({ operatorLogin, authRole, isObserver }) => (
			<InspectionsPanel
				operatorLogin={operatorLogin}
				authRole={authRole}
				isObserver={isObserver}
			/>
		),
		recommendations: ({ operatorLogin, authRole, isObserver }) => (
			<RecommendationsPanel
				operatorLogin={operatorLogin}
				authRole={authRole}
				isObserver={isObserver}
			/>
		),
		binding_decisions: ({ operatorLogin, authRole, isObserver }) => (
			<ObligatingDecisionsPanel
				operatorLogin={operatorLogin}
				authRole={authRole}
				isObserver={isObserver}
			/>
		),
		sanction_requests: ({ operatorLogin, authRole, isObserver }) => (
			<SanctionRequestsPanel
				operatorLogin={operatorLogin}
				authRole={authRole}
				isObserver={isObserver}
			/>
		),
		reports: ({ operatorLogin }) => (
			<ReportsPanel operatorLogin={operatorLogin} />
		),
		report_protocol_time: ({ operatorLogin }) => (
			<ProtocolTimePanel operatorLogin={operatorLogin} />
		),
		report_statement_time: ({ operatorLogin }) => (
			<StatementTimePanel operatorLogin={operatorLogin} />
		),
		report_own_time: ({ operatorLogin }) => (
			<OwnTimeReportPanel operatorLogin={operatorLogin} />
		),
	},
	management: {
		dictionaries: ({ operatorLogin, canEditDictionaries, authRole }) => (
			<DictionariesSheetPanel
				key="dictionaries"
				operatorLogin={operatorLogin}
				authRole={authRole}
				title="Słowniki"
				subtitle=""
				canEdit={canEditDictionaries}
			/>
		),
		teams: ({ operatorLogin, authRole }) => (
			<DictionariesSheetPanel
				key="teams"
				operatorLogin={operatorLogin}
				authRole={authRole}
				preferredKodTypu="zespoly"
				tableOnly
				title="Zespoły"
				subtitle=""
			/>
		),
		users: ({ operatorLogin, authRole }) => (
			<UsersPanel operatorLogin={operatorLogin} authRole={authRole} />
		),
		external_users: ({ operatorLogin, authRole }) => (
			<UsersPanel operatorLogin={operatorLogin} authRole={authRole} />
		),
		schedules: ({ operatorLogin, authRole }) => (
			<SchedulesPanel operatorLogin={operatorLogin} authRole={authRole} />
		),
		logs: ({ operatorLogin }) => <LogsPanel operatorLogin={operatorLogin} />,
	},
};

export function renderTabContent({
	panelMode,
	selectedItemId,
	operatorLogin,
	authRole,
	isObserver,
	canEditDictionaries,
}: RenderTabContentParams) {
	const renderer = TAB_CONTENT_RENDERERS[panelMode]?.[selectedItemId];
	if (!renderer) {
		return <PlaceholderPanel />;
	}

	return renderer({
		panelMode,
		selectedItemId,
		operatorLogin,
		authRole,
		isObserver,
		canEditDictionaries,
	});
}
