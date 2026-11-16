import { memo } from 'react';
import { Pressable, Text } from 'react-native';

import { GameCover } from '@/components/GameCover';
import { formatLastPlayed, type Game } from '@/lib/library';

export const COVER_RATIO = 4 / 3;

type GameGridCardProps = {
  game: Game;
  width: number;
  showLastPlayed?: boolean;
  onPress?: (game: Game) => void;
  onLongPress?: (game: Game) => void;
};

// Memoized so scrolling the library only renders cards that come into view, not every card each
// time the list re-renders.
export const GameGridCard = memo(function GameGridCard({
  game,
  width,
  showLastPlayed = true,
  onPress,
  onLongPress,
}: GameGridCardProps) {
  const hint = game.favorite
    ? 'Long press to remove from favorites'
    : 'Long press to add to favorites';

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={game.title}
      accessibilityHint={onLongPress ? hint : undefined}
      disabled={!onPress && !onLongPress}
      onPress={onPress && (() => onPress(game))}
      onLongPress={onLongPress && (() => onLongPress(game))}
      style={{ width }}
      className="mb-5 active:opacity-70">
      <GameCover game={game} width={width} height={width * COVER_RATIO} showDetails />
      <Text className="mt-2 font-inter-semibold text-[14px] text-foreground" numberOfLines={1}>
        {game.title}
      </Text>
      {showLastPlayed && (
        <Text className="mt-0.5 font-inter text-[12px] text-secondary" numberOfLines={1}>
          {formatLastPlayed(game.lastPlayedAt)}
        </Text>
      )}
    </Pressable>
  );
});
