import { loadSettings, saveSettings } from '@/lib/setup';
import KrsEmulator from '@/modules/krs-emulator';

export type MappingAction =
  | 'A'
  | 'B'
  | 'X'
  | 'Y'
  | 'L'
  | 'R'
  | 'ZL'
  | 'ZR'
  | 'START'
  | 'SELECT'
  | 'FAST_FORWARD'
  | 'MENU'
  | 'QUICK_SAVE'
  | 'QUICK_LOAD';

/** The keys that must be held together for each action; empty means not set. */
export type ControllerMapping = Record<MappingAction, string[]>;

export const DEFAULT_CONTROLLER_MAPPING: ControllerMapping = {
  A: ['BUTTON_A'],
  B: ['BUTTON_B'],
  X: ['BUTTON_X'],
  Y: ['BUTTON_Y'],
  L: ['BUTTON_L1'],
  R: ['BUTTON_R1'],
  // The triggers, which fast-forward also uses; a running 3DS game takes them first.
  ZL: ['BUTTON_L2'],
  ZR: ['BUTTON_R2'],
  START: ['BUTTON_START'],
  SELECT: ['BUTTON_SELECT'],
  FAST_FORWARD: ['BUTTON_R2'],
  MENU: ['BUTTON_MODE'],
  QUICK_SAVE: ['BUTTON_SELECT', 'BUTTON_R1'],
  QUICK_LOAD: [],
};

export type MappingRowInfo = {
  action: MappingAction;
  label: string;
  /** Which systems use the button, when not all of them do. */
  systems?: string;
};

export const BUTTON_ROWS: MappingRowInfo[] = [
  { action: 'A', label: 'A' },
  { action: 'B', label: 'B' },
  { action: 'X', label: 'X', systems: 'DS · 3DS' },
  { action: 'Y', label: 'Y', systems: 'DS · 3DS' },
  { action: 'L', label: 'L' },
  { action: 'R', label: 'R' },
  { action: 'ZL', label: 'ZL', systems: '3DS' },
  { action: 'ZR', label: 'ZR', systems: '3DS' },
  { action: 'START', label: 'Start' },
  { action: 'SELECT', label: 'Select' },
];

export const SHORTCUT_ROWS: MappingRowInfo[] = [
  { action: 'FAST_FORWARD', label: 'Fast-forward (hold)' },
  { action: 'MENU', label: 'Open menu' },
  { action: 'QUICK_SAVE', label: 'Quick save' },
  { action: 'QUICK_LOAD', label: 'Quick load' },
];

// Mappings saved before DS and 3DS support have no X, Y, ZL or ZR, so they're filled in from the defaults.
export const loadControllerMapping = (): ControllerMapping => ({
  ...DEFAULT_CONTROLLER_MAPPING,
  ...loadSettings().controllerMapping,
});

export async function saveControllerMapping(mapping: ControllerMapping) {
  KrsEmulator.setControllerMapping(JSON.stringify(mapping));
  await saveSettings({ controllerMapping: mapping });
}

// ZL and ZR may share keys with fast-forward: a 3DS game gets them, other systems fast-forward.
const SHARES_WITH_FAST_FORWARD: MappingAction[] = ['ZL', 'ZR'];

const canShare = (a: MappingAction, b: MappingAction) =>
  (a === 'FAST_FORWARD' && SHARES_WITH_FAST_FORWARD.includes(b)) ||
  (b === 'FAST_FORWARD' && SHARES_WITH_FAST_FORWARD.includes(a));

/** Binds keys to an action, clearing any other action that used exactly those keys. */
export function assignKeys(mapping: ControllerMapping, action: MappingAction, keys: string[]) {
  const wanted = [...keys].sort().join('+');
  const next = { ...mapping };
  for (const other of Object.keys(next) as MappingAction[]) {
    if (other === action || canShare(action, other)) continue;
    if ([...next[other]].sort().join('+') === wanted) next[other] = [];
  }
  next[action] = keys;
  return next;
}

export function formatKeys(keys: string[]) {
  return keys.map((key) => (keys.length > 1 ? key.replace('BUTTON_', '') : key)).join(' + ');
}
