import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')

print("АНАЛИЗ ДАННЫХ LEAGUE OF LEGENDS")

# 1. АНАЛИЗ ВЫБОРА/БАНОВ ЧЕМПИОНОВ
print("")
print("")
print("1. АНАЛИЗ ВЫБОРА И БАНОВ ЧЕМПИОНОВ")

def analyze_champions_simple():
    """Анализ выбора и банов чемпионов"""
    try:
        import kagglehub
        
        # Загрузка датасета
        path = kagglehub.dataset_download("paololol/league-of-legends-ranked-matches")
        
        # 1. Загружаем champs.csv (справочник чемпионов)
        champs_df = pd.read_csv(os.path.join(path, 'champs.csv'))
        print(f"Загружено {champs_df.shape[0]} чемпионов")
        
        # Создаем словарь для маппинга ID чемпионов на имена
        champ_id_to_name = dict(zip(champs_df['id'], champs_df['name']))
        
        # 2. Загружаем teambans.csv (баны)
        teambans_df = pd.read_csv(os.path.join(path, 'teambans.csv'))
        print(f"Загружено {teambans_df.shape[0]:,} записей о банах")
        
        # 3. Загружаем participants.csv (участники)
        participants_df = pd.read_csv(os.path.join(path, 'participants.csv'))
        print(f"Загружено {participants_df.shape[0]:,} записей об участниках")
        
        # АНАЛИЗ БАНОВ ЧЕМПИОНОВ
        print("\n" + "-"*50)
        print("АНАЛИЗ БАНОВ ЧЕМПИОНОВ:")
        
        # Подсчитываем частоту банов каждого чемпиона
        ban_counts = teambans_df['championid'].value_counts().head(15)
        
        # Преобразуем ID в имена
        ban_counts_named = pd.Series(ban_counts.values, 
                                     index=[champ_id_to_name.get(idx, f'ID_{idx}') for idx in ban_counts.index])
        
        # Создаем DataFrame для отображения
        total_matches_teambans = teambans_df['matchid'].nunique()
        ban_df = pd.DataFrame({
            'Чемпион': ban_counts_named.index,
            'Количество банов': ban_counts_named.values,
            'Процент матчей': [round(val / total_matches_teambans * 100, 1) for val in ban_counts_named.values]
        })
        
        print("\nТоп-15 самых часто банимых чемпионов:")
        print(ban_df.to_string(index=False))
        
        # АНАЛИЗ ВЫБОРА ЧЕМПИОНОВ
        print("\n" + "-"*50)
        print("АНАЛИЗ ВЫБОРА ЧЕМПИОНОВ:")
        
        # Подсчитываем частоту выбора каждого чемпиона
        pick_counts = participants_df['championid'].value_counts().head(15)
        
        # Преобразуем ID в имена
        pick_counts_named = pd.Series(pick_counts.values,
                                      index=[champ_id_to_name.get(idx, f'ID_{idx}') for idx in pick_counts.index])
        
        # Создаем DataFrame для отображения
        total_matches_participants = participants_df['matchid'].nunique()
        pick_df = pd.DataFrame({
            'Чемпион': pick_counts_named.index,
            'Количество выборов': pick_counts_named.values,
            'Процент матчей': [round(val / (total_matches_participants * 10) * 100, 1) for val in pick_counts_named.values]
        })
        
        print("\nТоп-15 самых часто выбираемых чемпионов:")
        print(pick_df.to_string(index=False))
        
        # ВИЗУАЛИЗАЦИЯ
        print("")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        print("")
        
        # 1. Топ банов (горизонтальная диаграмма)
        ax1 = axes[0, 0]
        ban_counts_named.head(10).plot(kind='barh', ax=ax1, color='lightcoral')
        ax1.set_xlabel('Количество банов')
        ax1.set_title('Топ-10 самых часто банимых чемпионов', fontsize=12)
        ax1.invert_yaxis()
        
        # 2. Топ пиков (горизонтальная диаграмма)
        ax2 = axes[0, 1]
        pick_counts_named.head(10).plot(kind='barh', ax=ax2, color='lightblue')
        ax2.set_xlabel('Количество выборов')
        ax2.set_title('Топ-10 самых часто выбираемых чемпионов', fontsize=12)
        ax2.invert_yaxis()
        
        # 3. Сравнение топ-10 чемпионов по банам и пикам
        ax3 = axes[1, 0]
        # Берем топ-10 чемпионов по сумме банов и пиков
        all_champs = {}
        for champ_id in set(list(ban_counts.index[:20]) + list(pick_counts.index[:20])):
            all_champs[champ_id] = ban_counts.get(champ_id, 0) + pick_counts.get(champ_id, 0)
        
        top_10_ids = sorted(all_champs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        champ_names = []
        ban_data = []
        pick_data = []
        
        for champ_id, _ in top_10_ids:
            champ_names.append(champ_id_to_name.get(champ_id, f'ID_{champ_id}'))
            ban_data.append(ban_counts.get(champ_id, 0))
            pick_data.append(pick_counts.get(champ_id, 0))
        
        x = np.arange(len(champ_names))
        width = 0.35
        
        ax3.bar(x - width/2, ban_data, width, label='Баны', color='lightcoral')
        ax3.bar(x + width/2, pick_data, width, label='Пики', color='lightblue')
        ax3.set_xticks(x)
        ax3.set_xticklabels(champ_names, rotation=45, ha='right')
        ax3.set_ylabel('Количество')
        ax3.set_title('Топ-10 чемпионов: сравнение банов и пиков', fontsize=12)
        ax3.legend()
        
        # 4. Круговая диаграмма популярности чемпионов
        ax4 = axes[1, 1]
        # Рассчитываем процентную долю в топ-10
        top_10_total = sum([ban_counts.get(id, 0) + pick_counts.get(id, 0) for id, _ in top_10_ids])
        percentages = []
        labels = []
        
        for champ_id, _ in top_10_ids:
            champ_name = champ_id_to_name.get(champ_id, f'ID_{champ_id}')
            total = ban_counts.get(champ_id, 0) + pick_counts.get(champ_id, 0)
            percentage = (total / top_10_total) * 100
            percentages.append(percentage)
            labels.append(f'{champ_name}\n{round(percentage, 1)}%')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(percentages)))
        ax4.pie(percentages, labels=labels, colors=colors, startangle=90)
        ax4.set_title('Распределение популярности топ-10 чемпионов', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('champions_analysis.png', dpi=100, bbox_inches='tight')
        print("Визуализации сохранены как 'champions_analysis.png'")
        
        # Вывод итоговой таблицы
        print("ИТОГОВАЯ ТАБЛИЦА: САМЫЕ ПОПУЛЯРНЫЕ ЧЕМПИОНЫ")
        
        # Создаем сводную таблицу
        summary_data = []
        for champ_id, _ in top_10_ids[:10]:
            champ_name = champ_id_to_name.get(champ_id, f'ID_{champ_id}')
            bans = ban_counts.get(champ_id, 0)
            picks = pick_counts.get(champ_id, 0)
            total = bans + picks
            ban_rate = round(bans / total_matches_teambans * 100, 1)
            pick_rate = round(picks / (total_matches_participants * 10) * 100, 1)
            
            summary_data.append({
                'Чемпион': champ_name,
                'Баны': bans,
                'Пики': picks,
                'Всего': total,
                'Бан-рейт (%)': ban_rate,
                'Пик-рейт (%)': pick_rate
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return {
            'ban_counts': ban_counts_named,
            'pick_counts': pick_counts_named,
            'summary_df': summary_df
        }
        
    except Exception as e:
        print(f"Ошибка при анализе чемпионов: {e}")
        import traceback
        traceback.print_exc()
        return None

# Запускаем анализ чемпионов
champion_data = analyze_champions_simple()

# 2. АНАЛИЗ СТАТИСТИКИ ИГРОКОВ
print("")
print("")
print("2. АНАЛИЗ ОСНОВНЫХ СТАТИСТИК ИГРОКОВ")

def analyze_player_stats_simple():
    """Упрощенный анализ статистики игроков с исправленным названием столбца CS"""
    try:
        import kagglehub
        
        path = kagglehub.dataset_download("paololol/league-of-legends-ranked-matches")
        
        # Загружаем только stats1.csv для анализа
        try:
            stats_df = pd.read_csv(os.path.join(path, 'stats1.csv'), nrows=50000)
        except:
            # Пробуем загрузить без ограничений, если файл меньше
            stats_df = pd.read_csv(os.path.join(path, 'stats1.csv'))
        print(f"Загружено {stats_df.shape[0]:,} записей статистики")
        
        # Показываем все столбцы для отладки
        print("\nДоступные столбцы в stats1.csv:")
        print(stats_df.columns.tolist())
        
        # АНАЛИЗ ОСНОВНЫХ СТАТИСТИК
        print("\n" + "-"*50)
        print("ОСНОВНЫЕ СТАТИСТИКИ ИГРОКОВ:")
        
        # Выбираем ключевые столбцы для анализа
        key_columns = ['kills', 'deaths', 'assists', 'totminionskilled', 
                      'wardsplaced', 'visionscore', 'champlvl', 'win']
        
        # Проверяем, какие столбцы есть в данных
        available_columns = [col for col in key_columns if col in stats_df.columns]
        
        if not available_columns:
            print("Ключевые столбцы не найдены в данных")
            # Пробуем найти другие столбцы
            available_columns = list(stats_df.columns[:5])
            print(f"Используем первые 5 доступных столбцов: {available_columns}")
        
        # Выводим основные статистики
        print(f"\nАнализ {len(available_columns)} ключевых показателей:")
        
        # Рассчитываем KDA
        if all(col in stats_df.columns for col in ['kills', 'deaths', 'assists']):
            stats_df['kda'] = (stats_df['kills'] + stats_df['assists']) / stats_df['deaths'].replace(0, 1)
            if 'kda' not in available_columns:
                available_columns.append('kda')
        
        # Основные статистики
        stats_summary = stats_df[available_columns].describe().round(2)
        
        print("\nОсновные статистики:")
        print(stats_summary)
        
        # Корреляция с победой
        if 'win' in stats_df.columns:
            print("\n" + "-"*50)
            print("КОРРЕЛЯЦИЯ СТАТИСТИК С ПОБЕДОЙ:")
            
            # Выбираем только числовые столбцы
            numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
            if 'win' in numeric_cols:
                win_correlations = stats_df[numeric_cols].corr()['win'].abs().sort_values(ascending=False)
                
                # Исключаем саму победу
                if 'win' in win_correlations.index:
                    win_correlations = win_correlations.drop('win')
                
                top_corrs = win_correlations.head(10)
                
                print("\nТоп-10 статистик, наиболее коррелирующих с победой:")
                for i, (stat, corr) in enumerate(top_corrs.items(), 1):
                    print(f"{i:2d}. {stat:25s}: {corr:.3f}")
        
        # ВИЗУАЛИЗАЦИЯ
        print("")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        print("")
        # 1. Распределение KDA
        ax1 = axes[0, 0]
        if 'kda' in stats_df.columns:
            # Ограничиваем KDA для лучшей визуализации
            kda_filtered = stats_df[stats_df['kda'] < 10]['kda']
            ax1.hist(kda_filtered, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
            ax1.set_xlabel('KDA')
            ax1.set_ylabel('Частота')
            ax1.set_title('Распределение KDA игроков', fontsize=12)
            ax1.axvline(stats_df['kda'].mean(), color='red', linestyle='--',
                       label=f'Среднее: {stats_df["kda"].mean():.2f}')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Нет данных о KDA', ha='center', va='center', fontsize=12)
        
        # 2. Распределение CS (Creep Score)
        ax2 = axes[0, 1]
        if 'totminionskilled' in stats_df.columns:
            ax2.hist(stats_df['totminionskilled'], bins=30, edgecolor='black', alpha=0.7, color='lightblue')
            ax2.set_xlabel('Количество миньонов (CS)')
            ax2.set_ylabel('Частота')
            ax2.set_title('Распределение CS за игру', fontsize=12)
            ax2.axvline(stats_df['totminionskilled'].mean(), color='red', linestyle='--',
                       label=f'Среднее: {stats_df["totminionskilled"].mean():.0f}')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Нет данных о CS', ha='center', va='center', fontsize=12)
        
        # 3. Распределение убийств
        ax3 = axes[0, 2]
        if 'kills' in stats_df.columns:
            ax3.hist(stats_df['kills'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
            ax3.set_xlabel('Количество убийств')
            ax3.set_ylabel('Частота')
            ax3.set_title('Распределение убийств за игру', fontsize=12)
            ax3.axvline(stats_df['kills'].mean(), color='red', linestyle='--',
                       label=f'Среднее: {stats_df["kills"].mean():.1f}')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Нет данных об убийствах', ha='center', va='center', fontsize=12)
        
        # 4. Зависимость KDA от уровня чемпиона
        ax4 = axes[1, 0]
        if 'kda' in stats_df.columns and 'champlvl' in stats_df.columns:
            sample_size = min(1000, len(stats_df))
            sample = stats_df.sample(sample_size, random_state=42)
            ax4.scatter(sample['champlvl'], sample['kda'], alpha=0.6, s=20, color='purple')
            ax4.set_xlabel('Уровень чемпиона')
            ax4.set_ylabel('KDA')
            ax4.set_title('Зависимость KDA от уровня чемпиона', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'Нет данных для графика', ha='center', va='center', fontsize=12)
        
        # 5. Корреляционная матрица ключевых статистик
        ax5 = axes[1, 1]
        if len(available_columns) > 3:
            # Выбираем статистики для корреляционной матрицы, исключая win
            corr_stats = [col for col in available_columns if col != 'win'][:6]
            if len(corr_stats) >= 3:
                corr_matrix = stats_df[corr_stats[:5]].corr()
                im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax5.set_xticks(range(len(corr_matrix.columns)))
                ax5.set_yticks(range(len(corr_matrix.columns)))
                ax5.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
                ax5.set_yticklabels(corr_matrix.columns)
                ax5.set_title('Корреляция ключевых статистик', fontsize=12)
                
                # Добавляем значения в ячейки
                for i in range(len(corr_matrix.columns)):
                    for j in range(len(corr_matrix.columns)):
                        ax5.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
                
                plt.colorbar(im, ax=ax5, shrink=0.8)
            else:
                ax5.text(0.5, 0.5, 'Недостаточно данных\nдля матрицы корреляций', 
                        ha='center', va='center', fontsize=12)
        else:
            ax5.text(0.5, 0.5, 'Недостаточно статистик\nдля анализа', 
                    ha='center', va='center', fontsize=12)
        
        # 6. Boxplot основных статистик
        ax6 = axes[1, 2]
        # Выбираем статистики для boxplot, исключая win
        boxplot_stats = [col for col in available_columns if col not in ['win', 'kda']][:4]
        if len(boxplot_stats) >= 3:
            stats_to_plot = stats_df[boxplot_stats].copy()
            
            # Ограничиваем выбросы для лучшей визуализации
            for col in boxplot_stats:
                if col in stats_to_plot.columns:
                    # Убираем крайние выбросы
                    q1 = stats_to_plot[col].quantile(0.25)
                    q3 = stats_to_plot[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    stats_to_plot.loc[stats_to_plot[col] > upper_bound, col] = upper_bound
            
            bp = ax6.boxplot([stats_to_plot[col].dropna() for col in boxplot_stats], 
                            labels=boxplot_stats, patch_artist=True)
            
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(boxplot_stats)]):
                patch.set_facecolor(color)
            
            ax6.set_ylabel('Значение')
            ax6.set_title('Распределение ключевых статистик', fontsize=12)
            ax6.tick_params(axis='x', rotation=45)
        else:
            ax6.text(0.5, 0.5, 'Недостаточно статистик\nдля boxplot', 
                    ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('player_stats_analysis.png', dpi=100, bbox_inches='tight')
        print("Визуализации сохранены как 'player_stats_analysis.png'")
        
        # Вывод итоговой таблицы статистик
        print("ИТОГОВАЯ ТАБЛИЦА: СРЕДНИЕ ЗНАЧЕНИЯ СТАТИСТИК")
        
        if 'win' in stats_df.columns:
            # Добавляем goldearned и totminionskilled в таблицу сравнения
            comparison_columns = available_columns.copy()
            
            # Добавляем goldearned если есть
            if 'goldearned' in stats_df.columns and 'goldearned' not in comparison_columns:
                comparison_columns.append('goldearned')
            
            # Добавляем totminionskilled если есть
            if 'totminionskilled' in stats_df.columns and 'totminionskilled' not in comparison_columns:
                comparison_columns.append('totminionskilled')
            
            # Сравнение статистик для побед и поражений
            win_stats = stats_df[stats_df['win'] == 1][comparison_columns].mean()
            lose_stats = stats_df[stats_df['win'] == 0][comparison_columns].mean()
            
            # Создаем понятные названия для столбцов
            display_names = []
            for col in comparison_columns:
                if col == 'totminionskilled':
                    display_names.append('CS (минньоны)')
                elif col == 'goldearned':
                    display_names.append('Золото')
                elif col == 'kda':
                    display_names.append('KDA')
                elif col == 'champlvl':
                    display_names.append('Уровень чемпиона')
                elif col == 'wardsplaced':
                    display_names.append('Установлено вардов')
                elif col == 'visionscore':
                    display_names.append('Видение')
                elif col == 'kills':
                    display_names.append('Убийства')
                elif col == 'deaths':
                    display_names.append('Смерти')
                elif col == 'assists':
                    display_names.append('Помощь')
                elif col == 'win':
                    display_names.append('Победа')
                else:
                    display_names.append(col)
            
            comparison_df = pd.DataFrame({
                'Статистика': display_names,
                'При победе': win_stats.values.round(2),
                'При поражении': lose_stats.values.round(2),
                'Разница': (win_stats.values - lose_stats.values).round(2)
            })
            
            print(comparison_df.to_string(index=False))
        else:
            # Просто средние значения
            # Создаем понятные названия для столбцов
            display_names = []
            for col in available_columns:
                if col == 'totminionskilled':
                    display_names.append('CS (минньоны)')
                elif col == 'goldearned':
                    display_names.append('Золото')
                elif col == 'kda':
                    display_names.append('KDA')
                elif col == 'champlvl':
                    display_names.append('Уровень чемпиона')
                elif col == 'wardsplaced':
                    display_names.append('Установлено вардов')
                elif col == 'visionscore':
                    display_names.append('Видение')
                elif col == 'kills':
                    display_names.append('Убийства')
                elif col == 'deaths':
                    display_names.append('Смерти')
                elif col == 'assists':
                    display_names.append('Помощь')
                elif col == 'win':
                    display_names.append('Победа')
                else:
                    display_names.append(col)
            
            mean_stats = pd.DataFrame({
                'Статистика': display_names,
                'Среднее значение': stats_df[available_columns].mean().values.round(2),
                'Медиана': stats_df[available_columns].median().values.round(2),
                'Стандартное отклонение': stats_df[available_columns].std().values.round(2)
            })
            
            print(mean_stats.to_string(index=False))
        
        return {
            'stats_summary': stats_summary,
            'stats_df': stats_df
        }
        
    except Exception as e:
        print(f"Ошибка при анализе статистики игроков: {e}")
        import traceback
        traceback.print_exc()
        return None

# Запускаем анализ статистики игроков
player_stats_data = analyze_player_stats_simple()

# 3. АНАЛИЗ МАТЧЕЙ И ДЛИТЕЛЬНОСТИ
print("")
print("")
print("3. АНАЛИЗ МАТЧЕЙ И ДЛИТЕЛЬНОСТИ")

def analyze_matches_simple():
    """Анализ матчей"""
    try:
        import kagglehub
        
        path = kagglehub.dataset_download("paololol/league-of-legends-ranked-matches")
        
        # Загружаем matches.csv
        try:
            matches_df = pd.read_csv(os.path.join(path, 'matches.csv'), nrows=20000)
        except:
            matches_df = pd.read_csv(os.path.join(path, 'matches.csv'))
        print(f"Загружено {matches_df.shape[0]:,} записей о матчах")
        
        # АНАЛИЗ ДЛИТЕЛЬНОСТИ МАТЧЕЙ
        print("\n" + "-"*50)
        print("АНАЛИЗ ДЛИТЕЛЬНОСТИ МАТЧЕЙ:")
        
        # Находим столбец с длительностью
        duration_col = None
        for col in matches_df.columns:
            if 'duration' in col.lower():
                duration_col = col
                break
        
        if duration_col:
            # Проверяем, в каких единицах измерения длительность
            avg_duration = matches_df[duration_col].mean()
            if avg_duration > 100:
                # Это секунды, переводим в минуты
                matches_df['duration_minutes'] = matches_df[duration_col] / 60
                print("Длительность переведена из секунд в минуты")
            else:
                matches_df['duration_minutes'] = matches_df[duration_col]
            
            # Основные статистики длительности
            duration_stats = matches_df['duration_minutes'].describe()
            
            print(f"\nОсновные статистики длительности матчей:")
            print(f"• Средняя длительность: {duration_stats['mean']:.1f} минут")
            print(f"• Медианная длительность: {duration_stats['50%']:.1f} минут")
            print(f"• Минимальная длительность: {duration_stats['min']:.1f} минут")
            print(f"• Максимальная длительность: {duration_stats['max']:.1f} минут")
            print(f"• Стандартное отклонение: {duration_stats['std']:.1f} минут")
            
            # Распределение по длительности
            print("\nРаспределение матчей по длительности:")
            bins = [0, 15, 25, 35, 45, float('inf')]
            labels = ['<15 мин', '15-25 мин', '25-35 мин', '35-45 мин', '>45 мин']
            
            matches_df['duration_group'] = pd.cut(matches_df['duration_minutes'], bins=bins, labels=labels)
            duration_dist = matches_df['duration_group'].value_counts().sort_index()
            
            for group, count in duration_dist.items():
                percentage = (count / len(matches_df)) * 100
                print(f"• {group}: {count} матчей ({percentage:.1f}%)")
        else:
            print("Столбец с длительностью не найден")
        
        # Анализ платформ
        platform_col = None
        for col in matches_df.columns:
            if 'platform' in col.lower():
                platform_col = col
                break
        
        if platform_col:
            print("\n" + "-"*50)
            print("РАСПРЕДЕЛЕНИЕ МАТЧЕЙ ПО ПЛАТФОРМАМ:")
            
            platform_dist = matches_df[platform_col].value_counts().head(10)
            
            print("\nТоп-10 платформ по количеству матчей:")
            for platform, count in platform_dist.items():
                percentage = (count / len(matches_df)) * 100
                print(f"• {platform}: {count} матчей ({percentage:.1f}%)")
        
        # ВИЗУАЛИЗАЦИЯ
        print("")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        print("")
        
        # 1. Распределение длительности матчей
        ax1 = axes[0, 0]
        if duration_col and 'duration_minutes' in matches_df.columns:
            # Фильтруем выбросы для лучшей визуализации
            duration_filtered = matches_df['duration_minutes']
            # Ограничиваем до 60 минут для визуализации
            duration_filtered = duration_filtered[duration_filtered <= 60]
            
            ax1.hist(duration_filtered, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax1.set_xlabel('Длительность (минуты)')
            ax1.set_ylabel('Количество матчей')
            ax1.set_title('Распределение длительности матчей', fontsize=12)
            ax1.axvline(matches_df['duration_minutes'].mean(), color='red', linestyle='--',
                       label=f'Среднее: {matches_df["duration_minutes"].mean():.1f} мин')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'Нет данных о длительности', ha='center', va='center', fontsize=12)
        
        # 2. Boxplot длительности
        ax2 = axes[0, 1]
        if duration_col and 'duration_minutes' in matches_df.columns:
            # Фильтруем выбросы
            duration_filtered = matches_df['duration_minutes']
            duration_filtered = duration_filtered[duration_filtered <= 60]
            
            ax2.boxplot(duration_filtered.dropna(), vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
            ax2.set_xlabel('Длительность (минуты)')
            ax2.set_title('Длительности матчей', fontsize=12)
            ax2.set_yticks([])
        else:
            ax2.text(0.5, 0.5, 'Нет данных о длительности', ha='center', va='center', fontsize=12)
        
        # 3. Распределение по платформам
        ax3 = axes[1, 0]
        if platform_col:
            platform_counts = matches_df[platform_col].value_counts().head(8)
            ax3.bar(range(len(platform_counts)), platform_counts.values, color='lightgreen')
            ax3.set_xticks(range(len(platform_counts)))
            ax3.set_xticklabels(platform_counts.index, rotation=45, ha='right')
            ax3.set_xlabel('Платформа')
            ax3.set_ylabel('Количество матчей')
            ax3.set_title('Топ платформ по матчам', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'Нет данных о платформах', ha='center', va='center', fontsize=12)
        
        # 4. Круговая диаграмма распределения по длительности
        ax4 = axes[1, 1]
        if duration_col and 'duration_group' in matches_df.columns:
            group_counts = matches_df['duration_group'].value_counts()
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(group_counts)))
            ax4.pie(group_counts.values, labels=group_counts.index, colors=colors, autopct='%1.1f%%')
            ax4.set_title('Распределение матчей по длительности', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'Нет данных для диаграммы', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('matches_analysis.png', dpi=100, bbox_inches='tight')
        print("Визуализации сохранены как 'matches_analysis.png'")
        
        # Вывод итоговой таблицы
        print("ИТОГОВАЯ ТАБЛИЦА: СТАТИСТИКИ МАТЧЕЙ")
        
        summary_data = []
        
        if duration_col and 'duration_group' in matches_df.columns:
            # Статистики длительности по группам
            for group in labels:
                group_data = matches_df[matches_df['duration_group'] == group]
                if len(group_data) > 0:
                    avg_duration = group_data['duration_minutes'].mean()
                    summary_data.append({
                        'Группа длительности': group,
                        'Количество матчей': len(group_data),
                        'Процент': round(len(group_data) / len(matches_df) * 100, 1),
                        'Средняя длительность': round(avg_duration, 1)
                    })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            print(summary_df.to_string(index=False))
        
        return {
            'matches_df': matches_df,
            'duration_stats': duration_stats if duration_col else None
        }
        
    except Exception as e:
        print(f"Ошибка при анализе матчей: {e}")
        import traceback
        traceback.print_exc()
        return None

# Запускаем анализ матчей
matches_data = analyze_matches_simple()

# 4. АНАЛИЗ ЗАВИСИМОСТИ ВИНРЕЙТА ОТ РАЗНИЦЫ В ЗОЛОТЕ
print("")
print("")
print("4. АНАЛИЗ ЗАВИСИМОСТИ ВИНРЕЙТА ОТ РАЗНИЦЫ В ЗОЛОТЕ КОМАНД")

def analyze_gold_difference_winrate():
    """Анализ зависимости винрейта от разницы в золоте между командами"""
    try:
        import kagglehub
        
        path = kagglehub.dataset_download("paololol/league-of-legends-ranked-matches")
        
        # Загружаем stats1.csv и participants.csv

        try:
            stats_df = pd.read_csv(os.path.join(path, 'stats1.csv'), nrows=50000)
        except:
            stats_df = pd.read_csv(os.path.join(path, 'stats1.csv'))
        
        try:
            participants_df = pd.read_csv(os.path.join(path, 'participants.csv'), nrows=50000)
        except:
            participants_df = pd.read_csv(os.path.join(path, 'participants.csv'))
        
        print(f"Столбцы в stats1.csv: {list(stats_df.columns)[:10]}...")
        print(f"Столбцы в participants.csv: {list(participants_df.columns)}")
        
        # Проверяем доступные столбцы
        gold_columns = []
        for col in stats_df.columns:
            if 'gold' in col.lower():
                gold_columns.append(col)
        
        print(f"Найдены столбцы с золотом: {gold_columns}")
        
        # Используем goldearned - заработанное золото
        gold_col = 'goldearned' if 'goldearned' in stats_df.columns else gold_columns[0] if gold_columns else None
        
        if not gold_col:
            print("Столбцы с информацией о золоте не найдены!")
            return analyze_simplified_gold_analysis(stats_df, participants_df)
        
        print(f"Используем столбец: {gold_col}")
        
        # Ищем столбец с победой
        win_col = 'win' if 'win' in stats_df.columns else None
        
        if not win_col:
            print("Столбец с информацией о победе не найдены!")
            return None
        
        print(f"Используем столбец победы: {win_col}")
        
        # Проверяем общий столбец для соединения
        common_columns = set(stats_df.columns).intersection(set(participants_df.columns))
        print(f"Общие столбцы для соединения: {common_columns}")
        
        # Используем 'id' для соединения
        if 'id' in common_columns:
            merged_data = pd.merge(
                stats_df[['id', gold_col, win_col]],
                participants_df[['id', 'matchid', 'player']],
                on='id',
                how='inner'
            )
        else:
            return analyze_simplified_gold_analysis(stats_df, participants_df)
        
        print(f"Объединено {len(merged_data)} записей")
        
        if merged_data.empty:
            print("Объединенные данные пусты!")
            return None
        
        # Переименовываем столбцы для удобства
        merged_data = merged_data.rename(columns={
            gold_col: 'gold',
            win_col: 'win'
        })
        
        # Определяем команду на основе номера игрока
        merged_data['teamid'] = merged_data['player'].apply(
            lambda x: 100 if x <= 5 else 200
        )
        
        # Группируем по матчам и командам для подсчета общего золота
        team_gold = merged_data.groupby(['matchid', 'teamid']).agg({
            'gold': 'sum',
            'win': 'first'  # win одинаков для всех в команде
        }).reset_index()
        
        # Разделяем данные по командам (100 и 200)
        team_100 = team_gold[team_gold['teamid'] == 100][['matchid', 'gold', 'win']]
        team_200 = team_gold[team_gold['teamid'] == 200][['matchid', 'gold', 'win']]
        
        # Объединяем в одну таблицу
        gold_comparison = pd.merge(
            team_100.rename(columns={'gold': 'gold_100', 'win': 'win_100'}),
            team_200.rename(columns={'gold': 'gold_200', 'win': 'win_200'}),
            on='matchid',
            how='inner'
        )
        
        # Удаляем строки с NaN
        gold_comparison = gold_comparison.dropna()
        
        if gold_comparison.empty:
            print("Нет данных для сравнения золота между командами")
            return None
        
        
        # Вычисляем разницу в золоте
        gold_comparison['gold_diff'] = gold_comparison['gold_100'] - gold_comparison['gold_200']
        gold_comparison['gold_diff_abs'] = abs(gold_comparison['gold_diff'])
        
        # Определяем, какая команда выиграла
        gold_comparison['winning_team'] = np.where(gold_comparison['win_100'] == 1, 100, 200)
        gold_comparison['win_by_gold_diff'] = np.where(gold_comparison['gold_diff'] > 0, 100, 200)
        
        # Создаем бины для анализа
        max_gold_diff = gold_comparison['gold_diff'].max()
        min_gold_diff = gold_comparison['gold_diff'].min()
        
        # Динамически создаем бины на основе данных
        bin_count = 12
        bin_edges = np.linspace(min_gold_diff, max_gold_diff, bin_count + 1)
        bin_labels = []
        
        for i in range(len(bin_edges) - 1):
            bin_labels.append(f"{int(bin_edges[i])//1000}k-{int(bin_edges[i+1])//1000}k")
        
        gold_comparison['gold_diff_bin'] = pd.cut(gold_comparison['gold_diff'], bins=bin_edges, labels=bin_labels)
        
        # Группируем по бинам и считаем винрейт команды 100
        winrate_by_bin = gold_comparison.groupby('gold_diff_bin').agg({
            'win_100': 'mean',
            'matchid': 'count'
        }).reset_index()
        
        winrate_by_bin.columns = ['Разница в золоте', 'Винрейт команды 1', 'Количество матчей']
        winrate_by_bin['Винрейт команды 100'] = (winrate_by_bin['Винрейт команды 1'] * 100).round(1)
        
        print("\n" + "-"*50)
        print("ЗАВИСИМОСТЬ ВИНРЕЙТА ОТ РАЗНИЦЫ В ЗОЛОТЕ:")
        
        print("\nВинрейт команды 1 в зависимости от разницы в золоте:")
        print(winrate_by_bin.to_string(index=False))
        
        # Общая статистика
        print("\n" + "-"*50)
        print("ОБЩАЯ СТАТИСТИКА:")
        
        total_matches = len(gold_comparison)
        matches_where_richer_wins = len(gold_comparison[gold_comparison['winning_team'] == gold_comparison['win_by_gold_diff']])
        winrate_when_richer = (matches_where_richer_wins / total_matches * 100)
        
        print(f"• Средняя разница в золоте: {gold_comparison['gold_diff'].mean():.0f}")
        print(f"• Медианная разница в золоте: {gold_comparison['gold_diff'].median():.0f}")
        print(f"• Команда с большим золотом выигрывает в {winrate_when_richer:.1f}% случаев")
        
        # Корреляция между разницей в золоте и победой
        correlation = gold_comparison['gold_diff'].corr(gold_comparison['win_100'])
        print(f"• Корреляция между разницей в золоте и победой: {correlation:.3f}")
        
        # ВИЗУАЛИЗАЦИЯ
        print("")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        print("")
        
        # 1. Винрейт в зависимости от разницы в золоте
        ax1 = axes[0, 0]
        # Ограничиваем количество отображаемых бинов для читаемости
        display_bins = winrate_by_bin.head(12)
        x_pos = np.arange(len(display_bins))
        bars = ax1.bar(x_pos, display_bins['Винрейт команды 100'], color='skyblue', edgecolor='black')
        ax1.set_xlabel('Разница в золоте (команда 100 - команда 200)')
        ax1.set_ylabel('Винрейт команды 100 (%)')
        ax1.set_title('Зависимость винрейта от разницы в золоте', fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(display_bins['Разница в золоте'], rotation=45, ha='right')
        
        # Добавляем значения над столбцами
        for bar, count in zip(bars, display_bins['Количество матчей']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom', fontsize=8)
        
        # 2. Распределение разницы в золоте
        ax2 = axes[0, 1]
        ax2.hist(gold_comparison['gold_diff'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Разница в золоте')
        ax2.set_ylabel('Количество матчей')
        ax2.set_title('Распределение разницы в золоте между командами', fontsize=12)
        ax2.axvline(0, color='red', linestyle='--', label='Равенство')
        ax2.legend()
        
        # 3. Зависимость винрейта от абсолютной разницы в золоте
        ax3 = axes[1, 0]
        # Группируем по абсолютной разнице (в тыс. золота)
        gold_comparison['gold_diff_k'] = gold_comparison['gold_diff_abs'] / 1000
        
        bins_abs = [0, 5, 10, 15, 20, 30, 50, float('inf')]
        labels_abs = ['0-5k', '5-10k', '10-15k', '15-20k', '20-30k', '30-50k', '>50k']
        
        gold_comparison['gold_diff_abs_bin'] = pd.cut(gold_comparison['gold_diff_k'], 
                                                     bins=bins_abs, 
                                                     labels=labels_abs)
        
        winrate_by_abs_diff = gold_comparison.groupby('gold_diff_abs_bin').agg({
            'win_100': 'mean',
            'matchid': 'count'
        }).reset_index()
        
        winrate_by_abs_diff['win_100'] = winrate_by_abs_diff['win_100'] * 100
        
        x_pos2 = np.arange(len(winrate_by_abs_diff))
        bars2 = ax3.bar(x_pos2, winrate_by_abs_diff['win_100'], color='orange', edgecolor='black')
        ax3.set_xlabel('Абсолютная разница в золоте (тыс.)')
        ax3.set_ylabel('Винрейт команды с преимуществом (%)')
        ax3.set_title('Винрейт в зависимости от абсолютной разницы в золоте', fontsize=12)
        ax3.set_xticks(x_pos2)
        ax3.set_xticklabels(winrate_by_abs_diff['gold_diff_abs_bin'], rotation=45, ha='right')
        
        # Добавляем количество матчей
        for bar, count in zip(bars2, winrate_by_abs_diff['matchid']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{count}', ha='center', va='bottom', fontsize=8)
        
        # 4. Точечная диаграмма: разница в золоте vs результат
        ax4 = axes[1, 1]
        sample_size = min(500, len(gold_comparison))
        sample = gold_comparison.sample(sample_size, random_state=42)
        colors = ['red' if win == 0 else 'green' for win in sample['win_100']]
        ax4.scatter(sample['gold_diff'], sample['win_100'], alpha=0.6, s=30, c=colors)
        ax4.set_xlabel('Разница в золоте')
        ax4.set_ylabel('Победа команды 100 (0/1)')
        ax4.set_title('Зависимость победы от разницы в золоте', fontsize=12)
        ax4.axvline(0, color='blue', linestyle='--', alpha=0.5)
        
        # Добавляем линию тренда
        if len(sample) > 1:
            z = np.polyfit(sample['gold_diff'], sample['win_100'], 1)
            p = np.poly1d(z)
            ax4.plot(sorted(sample['gold_diff']), p(sorted(sample['gold_diff'])), 
                    color='purple', linestyle='-', linewidth=2, alpha=0.8,
                    label=f'Тренд (наклон: {z[0]:.4f})')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('gold_difference_analysis.png', dpi=100, bbox_inches='tight')
        print("Визуализации сохранены как 'gold_difference_analysis.png'")
        
        # Создаем сводную таблицу
        print("ИТОГОВАЯ ТАБЛИЦА: КЛЮЧЕВЫЕ ВЫВОДЫ")
        
        summary_data = [
            {'Показатель': 'Средняя разница в золоте', 'Значение': f"{gold_comparison['gold_diff'].mean():.0f}"},
            {'Показатель': 'Медианная разница в золоте', 'Значение': f"{gold_comparison['gold_diff'].median():.0f}"},
            {'Показатель': 'Команда с преимуществом в золоте выигрывает', 'Значение': f"{winrate_when_richer:.1f}% матчей"},
            {'Показатель': 'Корреляция разницы в золоте с победой', 'Значение': f"{correlation:.3f}"}
        ]
        
        if len(gold_comparison) > 0:
            summary_data.extend([
                {'Показатель': 'Минимальная разница в золоте', 'Значение': f"{gold_comparison['gold_diff'].min():.0f}"},
                {'Показатель': 'Максимальная разница в золоте', 'Значение': f"{gold_comparison['gold_diff'].max():.0f}"},
                {'Показатель': 'Стандартное отклонение разницы', 'Значение': f"{gold_comparison['gold_diff'].std():.0f}"}
            ])
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return {
            'gold_comparison': gold_comparison,
            'winrate_by_bin': winrate_by_bin,
            'summary_stats': summary_df
        }
        
    except Exception as e:
        print(f"Ошибка при анализе зависимости винрейта от разницы в золоте: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            # Перезагружаем данные для анализа
            import kagglehub
            path = kagglehub.dataset_download("paololol/league-of-legends-ranked-matches")
            stats_df = pd.read_csv(os.path.join(path, 'stats1.csv'), nrows=50000)
            return analyze_simplified_gold_analysis(stats_df, None)
        except:
            return None

# Запускаем анализ зависимости винрейта от разницы в золоте
gold_analysis_data = analyze_gold_difference_winrate()

# ФИНАЛЬНЫЙ СВОДНЫЙ ОТЧЕТ
print("ФИНАЛЬНЫЙ СВОДНЫЙ ОТЧЕТ")

print("\nОБЩАЯ СТАТИСТИКА АНАЛИЗА:")
print("-"*40)

# Статистика по чемпионам
if champion_data is not None and 'summary_df' in champion_data:
    print("")
    print("\n1. АНАЛИЗ ЧЕМПИОНОВ:")
    print(f"   • Проанализировано чемпионов: {len(champion_data['ban_counts']) if 'ban_counts' in champion_data else 'N/A'}")
    if 'ban_counts' in champion_data and len(champion_data['ban_counts']) > 0:
        print(f"   • Самый часто банимый: {champion_data['ban_counts'].index[0]}")
    if 'pick_counts' in champion_data and len(champion_data['pick_counts']) > 0:
        print(f"   • Самый часто выбираемый: {champion_data['pick_counts'].index[0]}")

# Статистика по игрокам
if player_stats_data is not None and 'stats_df' in player_stats_data:
    print("")
    print("\n2. АНАЛИЗ СТАТИСТИКИ ИГРОКОВ:")
    stats_df = player_stats_data['stats_df']
    if 'kda' in stats_df.columns:
        print(f"   • Средний KDA: {stats_df['kda'].mean():.2f}")
    if 'totalminionskilled' in stats_df.columns:
        print(f"   • Средний CS за игру: {stats_df['totalminionskilled'].mean():.0f}")

# Статистика по матчам
if matches_data is not None and 'duration_stats' in matches_data and matches_data['duration_stats'] is not None:
    print("")
    print("\n3. АНАЛИЗ МАТЧЕЙ:")
    print(f"   • Средняя длительность матча: {matches_data['duration_stats']['mean']:.1f} минут")
    print(f"   • Медианная длительность: {matches_data['duration_stats']['50%']:.1f} минут")

# Статистика по зависимости винрейта от разницы в золоте
if gold_analysis_data is not None and 'summary_stats' in gold_analysis_data:
    print("")
    print("\n4. АНАЛИЗ ЗАВИСИМОСТИ ВИНРЕЙТА ОТ РАЗНИЦЫ В ЗОЛОТЕ:")
    summary_stats = gold_analysis_data['summary_stats']
    for _, row in summary_stats.iterrows():
        print(f"   • {row['Показатель']}: {row['Значение']}")

print("\nСОЗДАННЫЕ ГРАФИКИ И ТАБЛИЦЫ:")
print("-"*40)
print("1. champions_analysis.png - анализ чемпионов)")
print("2. player_stats_analysis.png - статистика игроков)")
print("3. matches_analysis.png - анализ матчей")
print("4. gold_difference_analysis.png - зависимость винрейта от золота")

print("\nКЛЮЧЕВЫЕ ВЫВОДЫ:")
print("-"*40)

# Выводы на основе анализа
print("1. МЕТА-ЧЕМПИОНЫ:")
if champion_data is not None and 'summary_df' in champion_data and not champion_data['summary_df'].empty:
    top_champs = champion_data['summary_df'].head(3)
    print(f"   • Топ-3 самых популярных чемпиона:")
    for i in range(min(3, len(top_champs))):
        row = top_champs.iloc[i]
        print(f"     {i+1}. {row['Чемпион']} (банов: {row['Баны']}, пиков: {row['Пики']})")
else:
    print("   • Данные о чемпионах недоступны")

print("\n2. СТАТИСТИКИ ИГРОКОВ:")
if player_stats_data is not None and 'stats_df' in player_stats_data:
    stats_df = player_stats_data['stats_df']
    if all(col in stats_df.columns for col in ['kills', 'deaths', 'assists']):
        avg_kills = stats_df['kills'].mean()
        avg_deaths = stats_df['deaths'].mean()
        avg_assists = stats_df['assists'].mean()
        print(f"   • Средние показатели за игру: {avg_kills:.1f}/{avg_deaths:.1f}/{avg_assists:.1f} (K/D/A)")
    else:
        print("   • Недостаточно данных для расчета K/D/A")
else:
    print("   • Данные о статистике игроков недоступны")

print("\n3. ХАРАКТЕРИСТИКИ МАТЧЕЙ:")
if matches_data is not None and 'duration_stats' in matches_data and matches_data['duration_stats'] is not None:
    print(f"   • Типичная длительность матча: 25-35 минут")
    print(f"   • Большинство матчей ({matches_data['duration_stats']['50%']:.0f} мин) заканчиваются в средней игре")
else:
    print("   • Данные о матчах недоступны")

print("\n4. ЗАВИСИМОСТЬ ВИНРЕЙТА ОТ ЗОЛОТА:")
if gold_analysis_data is not None:
    print("   • Экономическое преимущество сильно коррелирует с победой")
    print("   • Команда с преимуществом в золоте выигрывает в большинстве случаев")
    if 'summary_stats' in gold_analysis_data:
        winrate_row = gold_analysis_data['summary_stats'][gold_analysis_data['summary_stats']['Показатель'] == 
                                                         'Команда с преимуществом в золоте выигрывает']
        if not winrate_row.empty:
            print(f"   • {winrate_row.iloc[0]['Значение']} матчей выигрывает команда с золотым преимуществом")

print("")
print("АНАЛИЗ ЗАВЕРШЕН!")
print("")

# Сохранение финального отчета
try:
    with open('final_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write("ФИНАЛЬНЫЙ ОТЧЕТ ПО АНАЛИЗУ LEAGUE OF LEGENDS\n")
        
        f.write("ВЫПОЛНЕННЫЕ АНАЛИЗЫ:\n")
        f.write("1. Анализ выбора и банов чемпионов\n")
        f.write("2. Анализ статистики игроков (без goldearned в визуализациях)\n")
        f.write("3. Анализ матчей и длительности\n")
        f.write("4. Анализ зависимости винрейта от разницы в золоте команд\n\n")

        f.write("")
        
        f.write("ОСНОВНЫЕ ВЫВОДЫ:\n")
        
        if champion_data is not None and 'summary_df' in champion_data and not champion_data['summary_df'].empty:
            top_champ = champion_data['summary_df'].iloc[0]['Чемпион']
            f.write(f"• Самый популярный чемпион: {top_champ}\n")
        else:
            f.write("• Данные о чемпионах недоступны\n")
        
        if player_stats_data is not None and 'stats_df' in player_stats_data:
            if 'kda' in player_stats_data['stats_df'].columns:
                avg_kda = player_stats_data['stats_df']['kda'].mean()
                f.write(f"• Средний KDA игроков: {avg_kda:.2f}\n")
            else:
                f.write("• Нет данных о KDA\n")
        else:
            f.write("• Данные о статистике игроков недоступны\n")
        
        if matches_data is not None and 'duration_stats' in matches_data and matches_data['duration_stats'] is not None:
            avg_duration = matches_data['duration_stats']['mean']
            f.write(f"• Средняя длительность матча: {avg_duration:.1f} минут\n")
        else:
            f.write("• Данные о матчах недоступны\n")
        
        if gold_analysis_data is not None and 'summary_stats' in gold_analysis_data:
            f.write("• Зависимость винрейта от золота: команда с преимуществом в золоте выигрывает в большинстве случаев\n\n")
        else:
            f.write("• Данные о зависимости винрейта от золота недоступны\n\n")

        f.write("")
        
        f.write("СОЗДАННЫЕ ФАЙЛЫ:\n")
        f.write("• champions_analysis.png\n")
        f.write("• player_stats_analysis.png\n")
        f.write("• matches_analysis.png\n")
        f.write("• gold_difference_analysis.png\n")
        f.write("• final_summary_report.txt\n")

        f.write("")
        
    print("Финальный отчет сохранен как 'final_summary_report.txt'")
except Exception as e:
    print(f"Ошибка при сохранении отчета: {e}")


# Показываем все графики
try:
    plt.show()
except Exception as e:
    print(f"Ошибка при отображении графиков: {e}")
